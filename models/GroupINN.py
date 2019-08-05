from . import tf, arguments, argparse
from functools import reduce

class gcn_classification_net:
    class loss_weights:
        cross_entropy = 1.0
        neg_penalty_reduce = 0.1
        neg_penalty_gnn = 0.2
        ortho_penalty_p = 0.2
        ortho_penalty_n = 0.2
        variance_penalty_p = 0.3
        variance_penalty_n = 0.5
        l2_penalty = 2e-3

    @classmethod
    def update_parser_argument(cls, parser: argparse.ArgumentParser):
        args, _ = parser.parse_known_args()

        parser.set_defaults(selected_model="gcn_classification_net")
        print("===> Selected model: GroupINN")

        group = parser.add_argument_group(title="GroupINN arguments")
        group.add_argument("--dropout_rate", default=0, type=float, help="(default: %(default)s)")
        group.add_argument("--c", default=0.85, type=float, help="(default: %(default)s)")
        group.add_argument("--feature_reduction", default=5, type=int, help="(default: %(default)s)")
        group.add_argument("--learning_rate", default=0.001, help="(default: %(default)s)")
        
        arguments.add_loss_weights_argument(parser, cls.loss_weights, cls.__name__)
        
        return parser

    def __init__(self):
        self.feature_notify = 0

    def runtime_init(self, features, labels, mode):
        self.losses = []
        self.is_training = (mode==tf.estimator.ModeKeys.TRAIN)

    def model_fn(self, features, labels,
        mode:tf.estimator.ModeKeys, params):
        """
        features: batch_features from input_fn
        labels: batch_labels from input_fn
        mode: An instance of tf.estimator.ModeKeys
        params: Additional configuration
        """
        self.runtime_init(features, labels, mode)

        # Load parameters
        self.num_features = params["args"].feature_reduction
        self.c = params["args"].c
        self.dropout_rate = params["args"].dropout_rate
        self.selected_timeseries = params["args"].selected_timeseries
        self.learning_rate = params["args"].learning_rate
        self.tf_summary = (not params["args"].no_tensorboard)

        # Construct network
        s_feature = features[self.selected_timeseries]

        s_feature_p = s_feature[0]
        s_feature_n = s_feature[1]

        num_columns = int(s_feature_p.shape[-1])

        self.initializer = tf.initializers.random_uniform(0, 0.5/self.num_features)

        p_reduce = self.dim_reduction(s_feature_p, self.num_features, "reduction_p",
            self.loss_weights.ortho_penalty_p, self.loss_weights.variance_penalty_p, self.loss_weights.neg_penalty_reduce)
        p_conv1 = self.gnn_conv(None, p_reduce, "conv1_p", self.loss_weights.neg_penalty_gnn)
        p_conv2 = self.gnn_conv(p_conv1, p_reduce, "conv2_p", self.loss_weights.neg_penalty_gnn)
        p_conv3 = self.gnn_conv(p_conv2, p_reduce, "conv3_p", self.loss_weights.neg_penalty_gnn)

        n_reduce = self.dim_reduction(s_feature_n, self.num_features, "reduction_n",
            self.loss_weights.ortho_penalty_n, self.loss_weights.variance_penalty_n, self.loss_weights.neg_penalty_reduce)
        n_conv1 = self.gnn_conv(None, n_reduce, "conv1_n", self.loss_weights.neg_penalty_gnn)
        n_conv2 = self.gnn_conv(n_conv1, n_reduce, "conv2_n", self.loss_weights.neg_penalty_gnn)
        n_conv3 = self.gnn_conv(n_conv2, n_reduce, "conv3_n", self.loss_weights.neg_penalty_gnn)

        conv_concat = tf.reshape(tf.concat([p_conv3,n_conv3], -1), [-1, 2*self.num_features**2])
        dense_output = self.dense_layers(conv_concat, self.loss_weights.l2_penalty)
        
        output = self.generate_output(dense_output, labels, mode)

        if self.is_training:
            if self.feature_notify % 10 == 0:
                print("Selected feature: {}".format(self.selected_timeseries))
                self.loss_weights._print_current_weights() #pylint: disable=E1101
                self.count_params()
            self.feature_notify += 1
        return output

    def dim_reduction(self, adj_matrix, num_reduce, name_scope,
            ortho_penalty, variance_penalty, neg_penalty):
        column_dim = int(adj_matrix.shape[-1])
        with tf.variable_scope(name_scope):
            kernel = tf.get_variable("dim_reduction_kernel", shape=[column_dim, num_reduce],
                trainable=True, initializer=self.initializer,
                regularizer=tf.contrib.layers.l1_regularizer(scale=0.05)
                )
            kernel_p = tf.nn.relu(kernel)
            AF = tf.tensordot(adj_matrix, kernel_p, axes=[[-1],[0]])
            reduced_adj_matrix = tf.transpose(
                    tf.tensordot(kernel_p, AF, axes=[[0],[1]]), #num_reduce*batch*num_reduce
                perm=[1,0,2], name="reduced_adj")

            if self.tf_summary:
                tf.summary.image("dim_reduction_kernel", tf.expand_dims(
                        tf.expand_dims(kernel, axis=0),
                    axis=-1))
                tf.summary.image("dim_reduction_kernel_p", tf.expand_dims(
                        tf.expand_dims(kernel_p, axis=0),
                    axis=-1))

            gram_matrix = tf.matmul(kernel_p, kernel_p, transpose_a=True)
            diag_elements = tf.diag_part(gram_matrix)
            zero = tf.constant(0, dtype=tf.float32)
            mask = tf.not_equal(diag_elements, zero)


            if ortho_penalty!=0:
                ortho_loss_matrix = tf.square(gram_matrix - tf.diag(diag_elements))
                ortho_loss = tf.multiply(ortho_penalty, tf.reduce_sum(ortho_loss_matrix), name="ortho_penalty")
                self.losses.append(ortho_loss)

            if variance_penalty!=0:
                _ , variance = tf.nn.moments(tf.boolean_mask(diag_elements,mask), axes=[0])
                variance_loss = tf.multiply(variance_penalty, variance, name="variance_penalty")
                self.losses.append(variance_loss)

            if neg_penalty!=0:
                neg_loss = tf.multiply(neg_penalty, tf.reduce_sum(tf.nn.relu(tf.constant(1e-6)-kernel)), name="negative_penalty")
                self.losses.append(neg_loss)

        return reduced_adj_matrix

    def gnn_conv(self, prev_output, adj_matrix, name_scope, neg_penalty): #I+c*A*X*W,X0=I
        feature_dim = int(adj_matrix.shape[-1])
        eye = tf.eye(feature_dim)
        with tf.variable_scope(name_scope):
            kernel = tf.get_variable("gnn_kernel",
                shape=[feature_dim,feature_dim], trainable=True, initializer=self.initializer)
            if prev_output is None:
                AXW = tf.tensordot(adj_matrix, kernel, [[-1],[0]])
            else:
                XW = tf.tensordot(prev_output, kernel, [[-1],[0]]) #batch*feature_dim*feature_dim
                AXW = tf.matmul(adj_matrix, XW)
            I_cAXW = eye+self.c*AXW
            y_relu = tf.nn.relu(I_cAXW)
            col_mean = tf.tile(tf.reduce_mean(y_relu, axis=-2, keepdims=True)+1e-6,[1,feature_dim,1])
            y_norm = tf.divide(y_relu, col_mean)
            output = tf.nn.softplus(y_norm, name="gnn_output")

            if self.tf_summary:
                tf.summary.image("gnn_kernel", tf.expand_dims(
                        tf.expand_dims(kernel, axis=0),
                    axis=-1))

            if neg_penalty!=0:
                neg_loss = tf.multiply(neg_penalty, tf.reduce_sum(tf.nn.relu(tf.constant(1e-6)-kernel)), name="negative_penalty")
                self.losses.append(neg_loss)

        return output

    def dense_layers(self, input_flat, l2_penalty, name_scope="dense_layers"):
        with tf.variable_scope(name_scope):
            output_layer = tf.layers.Dense(2, name="output_layer")
            logits = output_layer(input_flat)

            kernel_var = output_layer.trainable_variables[0]
            
            if l2_penalty != 0:
                dense_kernel = output_layer.trainable_variables[0].read_value()
                l2_loss = tf.multiply(l2_penalty, tf.nn.l2_loss(dense_kernel), name="l2_penalty")
                self.losses.append(l2_loss)
        return logits

    def generate_output(self, logits, labels, mode:tf.estimator.ModeKeys):
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits)
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Define loss function
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        self.losses.append(
            tf.multiply(self.loss_weights.cross_entropy,
                   tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits),
                   name="cross_entropy_loss")
        )
        # Define loss function
        loss = tf.reduce_sum(self.losses, name="total_loss")
        for loss_scalar in self.losses:
            tf.summary.scalar(loss_scalar.name, loss_scalar, family="loss")

        # Define accuracy metric
        eval_metric_ops = {
            "metrics/accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"]),
            "confusion_matrix/TP": tf.metrics.true_positives(
                labels=labels, predictions=predictions["classes"]),
            "confusion_matrix/TN": tf.metrics.true_negatives(
                labels=labels, predictions=predictions["classes"]),
            "confusion_matrix/FP": tf.metrics.false_positives(
                labels=labels, predictions=predictions["classes"]),
            "confusion_matrix/FN": tf.metrics.false_negatives(
                labels=labels, predictions=predictions["classes"]),
            "metrics/precision": tf.metrics.precision(
                labels=labels, predictions=predictions["classes"]),
            "metrics/recall": tf.metrics.recall(
                labels=labels, predictions=predictions["classes"])
            }
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def count_params(self):
        "print number of trainable variables"
        size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
        n = sum(size(v) for v in tf.trainable_variables())
        print("Model size: {}K".format(n / 1000))
