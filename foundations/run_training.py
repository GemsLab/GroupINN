from foundations import data_manager, arguments, Path, datetime, os, np, tf_utils, logging, tf
from itertools import chain
import scipy.io

def train_classifier(parser, network_cls):
    args = arguments.parse_args(parser)
    manager = arguments.config_manager_dict["Basic"].data_manager

    arguments.config_manager_dict["Basic"].set_model_dir()
    network = network_cls()

    for hook_func in arguments.function_hooks["model_setup"]:
        hook_func(args)
    
    HCP_classifier = tf.estimator.Estimator(
        model_fn=network.model_fn, model_dir=args.model_dir,
        config=arguments.config_manager_dict["TensorFlow"].estimator_config,
        params={"args": args})
    
    if args.restore_dir is not None:
        epoch_list = range(args.train_epoch + 1)
    else:
        epoch_list = range(1, args.train_epoch + 1)
    
    for i in epoch_list:
        if i > 0:
            print("===> Training epoch {}/{}".format(i, args.train_epoch))
            HCP_classifier.train(
                input_fn=manager.create_balanced_train_dataset)
            print("Training epoch finished. Evaluating...")
        else:
            print("===> Evaluating on restored model...")
        
        train_eval_result = HCP_classifier.evaluate(
            input_fn=lambda: manager.create_train_dataset(astest=True),
            name="train"
        )
        
        if args.validation_proportion:
            val_eval_result = HCP_classifier.evaluate(
                input_fn=manager.create_validation_dataset,
                name="val"
            )
        else:
            val_eval_result = None
        
        test_eval_result = HCP_classifier.evaluate(
            input_fn=manager.create_test_dataset,
            name="test"
        )
        
        epoch_name = (i if i > 0 else "restored")
        for hook_func in arguments.function_hooks["eval"]:
            hook_func(
                epoch_name=epoch_name,
                eval_result_dict={
                    "train": logging.strip_var_group(train_eval_result),
                    "validation": logging.strip_var_group(val_eval_result),
                    "test": logging.strip_var_group(test_eval_result)
                }
            )
        
        for hook_func in arguments.function_hooks["pred"]:
            hook_func(
                epoch_name=epoch_name, 
                pred_result_dict=manager.batch_test_wrapper(
                    estimator=HCP_classifier,
                    input_fn={
                        "train": lambda: manager.create_train_dataset(astest=True),
                        "validation": manager.create_validation_dataset if args.validation_proportion else None,
                        "test": manager.create_test_dataset
                    }
                )
            )

    post_train_dict = dict()
    for hook_func in arguments.function_hooks["post_train"]:
        hook_func(post_train_dict)
    
    return post_train_dict