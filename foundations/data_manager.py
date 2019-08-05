from . import np, Path, tf, sys
import pandas as pd
import scipy.io
from itertools import groupby, chain
from sklearn.model_selection import train_test_split

class data_manager:
    def __init__(self, args, train_data_repeat=1, reg_selected_features=None):
        # self.meta_filename = args.meta_filename
        self.meta_table = pd.read_csv(args.meta_filename)
        # self.quartile = quartile
        self.training_shuffle_random = np.random.RandomState( #pylint: disable=E1101
            seed=args.training_shuffle_seed)
        self.training_shuffle_randint = (
            lambda: self.training_shuffle_random.randint(sys.maxsize))
        self.train_data_repeat = train_data_repeat
        self.prefetch_batch_size = 2
        self.args = args

        train_mask = [True] * len(self.meta_table)
        test_mask = [True] * len(self.meta_table)

        data_selector = [self.train_test_selector]
        if self.args.ignore_missing_mat_level> 0:
            data_selector.append(self.mat_file_selector)

        for selector in data_selector:
            new_train_mask, new_test_mask = selector(train_mask, test_mask)
            train_mask = np.bitwise_and(train_mask, new_train_mask) #pylint: disable=E1101
            test_mask = np.bitwise_and(test_mask, new_test_mask) #pylint: disable=E1101

        self.train_filenames = self.meta_table[train_mask].Subject  # noqa: disable=E501
        self.test_filenames = self.meta_table[test_mask].Subject  # noqa: disable=E501

        self.train_labels = np.array(self.classification_label[train_mask]) #pylint: disable=E1101
        self.test_labels = self.classification_label[test_mask] #pylint: disable=E1101
        self.validation_filenames = None; self.validation_labels = None
        self.gen_validation_set()
        key_func = lambda x:x[1]
        self.train_filenames_grouped = {key: [x[0] for x in members]
            for key, members in groupby(
                    sorted(zip(self.train_filenames, self.train_labels), key=key_func)
                , key_func)}
        print("===> Training set size: {}; Validation set size: {}; Test set size: {}".format(
                len(self.train_filenames), 
                None if (self.validation_filenames is None) else len(self.validation_filenames), 
                len(self.test_filenames)
            ))
        self.num_folds = 1

    def train_test_selector(self, *args, **kwargs):
        assert not hasattr(self, "classification_label")
        self.classification_label = self.meta_table.Quartile
        
        train_mask = ((self.meta_table.Quartile != -1) & (self.meta_table.Train == 1))
        test_mask = ((self.meta_table.Quartile != -1) & (self.meta_table.Test == 1))
        
        return train_mask, test_mask

    def mat_file_selector(self, prev_train_mask, prev_test_mask):
        print("===> Scanning data path to check missing mat files")
        allow_partial = (self.args.ignore_missing_mat_level> 1)
        subjectID_list = self.meta_table.Subject

        train_mask = prev_train_mask
        test_mask = prev_test_mask
        for setname, mask in [("training", train_mask), ("testing", test_mask)]:
            for ind, subjectID in enumerate(subjectID_list):
                if mask[ind]:
                    s_file_exist = {key: self.mat_file_checker(subjectID, filename, checklevel=100, supresswarn=True) 
                        for key, filename in self.args.load_mat_dict.items()}
                    if not all(s_file_exist.values()):
                        if not any(s_file_exist.values()):
                            mask[ind] = False
                        elif not allow_partial:
                            mask[ind] = False
                        print(">>>>WARNING: Missing data for {}: {}{}".format(
                            subjectID, [key for key in s_file_exist if not s_file_exist[key]],
                            (" (removed from {} set)".format(setname) if not mask[ind] else "")))
        return train_mask, test_mask

    def gen_validation_set(self):
        if self.args.validation_proportion:
            train_names, validation_names, train_labels, validation_labels = train_test_split(
                    self.train_filenames, self.train_labels, 
                    test_size=self.args.validation_proportion,
                    random_state=self.args.validation_selection_seed,
                    stratify=self.train_labels
                )
            self.train_filenames = train_names
            self.train_labels = train_labels
            self.validation_filenames = validation_names
            self.validation_labels = validation_labels

    def gen_subject_tensor(self, subjectID):
        # Deal with load_mat_list: convert it to a dictionary
        load_mat_dict = self.args.load_mat_dict
        mat_data = {key: self.mat_file_loader(subjectID, filename) for key, filename in self.args.load_mat_dict.items()
            if self.mat_file_checker(subjectID, filename)}
        return list(mat_data.values())

    def load_dataset_map(self, dataset):
        feature_length = len(self.args.load_mat_dict)
        tensor_gen_f = lambda subjectID: tf.py_func(self.gen_subject_tensor,
            [subjectID], [tf.float64]*feature_length, stateful=False)
        data_size = [1200, 264]
        # print("Time series size: {}".format(data_size))
        # Set the shape of the tensor and then cast it to a supported type
        tensor_reshape_f = lambda tensors, label: \
            ({
                key: tf.cast(tf.reshape(next(tensors), data_size), tf.float32) for key in self.args.load_mat_dict
                }, label)
        f = lambda subjectID, subjectlabel: tensor_reshape_f(iter(tensor_gen_f(subjectID)), subjectlabel)
        return dataset.map(f, self.args.num_parallel_calls)

    def load_dataset(self, dataset):
        loaded_dataset = self.load_dataset_map(dataset)
        
    def create_train_dataset(self, astest=False):
        self.train_filename_dataset = tf.data.Dataset.from_tensor_slices((self.train_filenames, self.train_labels))
        if not astest:
            self.train_filename_dataset = self.train_filename_dataset.shuffle(
                buffer_size=len(self.train_filenames), seed=self.training_shuffle_randint())
            self.train_filename_dataset = self.train_filename_dataset.repeat(count=self.train_data_repeat)
        else:
            self.last_test_filenames = self.train_filenames
            self.last_test_labels = self.train_labels
        self.train_dataset = self.load_dataset(self.train_filename_dataset)
        self.train_dataset = self.train_dataset.batch(self.args.batch_size)
        self.train_dataset = self.train_dataset.prefetch(self.prefetch_batch_size)
        return self.train_dataset

    def create_balanced_train_dataset(self, astest=False):
        if astest:
            return self.create_train_dataset(astest=astest)

        # Construct filename dataset for each label
        self.train_filename_dataset_grouped = dict()
        max_len_dataset = max(map(len, self.train_filenames_grouped.values()))
        for label, filenames in self.train_filenames_grouped.items():
            label_vector = [label] * len(filenames)
            label_dataset = tf.data.Dataset.from_tensor_slices((filenames, label_vector))
            label_dataset = label_dataset.shuffle(
                buffer_size=len(filenames), seed=self.training_shuffle_randint())
            if len(filenames) < max_len_dataset:
                label_dataset = label_dataset.repeat()
            self.train_filename_dataset_grouped[label] = label_dataset

        for ind, dataset in enumerate(self.train_filename_dataset_grouped.values()):
            if ind==0:
                self.__combined_dataset = dataset
            else:
                self.__combined_dataset = tf.data.Dataset.zip(
                    (self.__combined_dataset, dataset)
                ).flat_map(
                    lambda x0, x1: tf.data.Dataset.from_tensors(x0).concatenate(
                        tf.data.Dataset.from_tensors(x1))
                )
        self.train_dataset = self.load_dataset(self.__combined_dataset)

        num_group = len(self.train_filename_dataset_grouped)
        if self.args.batch_size%num_group != 0:
            prev_batch_size = self.args.batch_size
            self.args.batch_size -= (self.args.batch_size%num_group)
            print("Batch sized should be multiple of number of classes ({}): changing from {} to {}".format(
                num_group, prev_batch_size, self.args.batch_size))
        self.train_dataset = self.train_dataset.batch(self.args.batch_size)
        self.train_dataset = self.train_dataset.prefetch(self.prefetch_batch_size)

        return self.train_dataset

    def create_validation_dataset(self, num_take=None):
        self.validation_filename_dataset = tf.data.Dataset.from_tensor_slices((self.validation_filenames, self.validation_labels))
        self.validation_dataset = self.load_dataset(self.validation_filename_dataset)
        self.validation_dataset = self.validation_dataset.batch(self.args.batch_size)
        self.validation_dataset = self.validation_dataset.prefetch(self.prefetch_batch_size)
        if num_take:
            self.last_test_filenames = self.validation_filenames[:num_take]
            self.last_test_labels = self.validation_labels[:num_take]
            return self.validation_dataset.take(num_take)
        else:
            self.last_test_filenames = self.validation_filenames
            self.last_test_labels = self.validation_labels
            return self.validation_dataset

    def create_test_dataset(self, num_take=None):
        self.test_filename_dataset = tf.data.Dataset.from_tensor_slices((self.test_filenames, self.test_labels))
        self.test_dataset = self.load_dataset(self.test_filename_dataset)
        self.test_dataset = self.test_dataset.batch(self.args.batch_size)
        self.test_dataset = self.test_dataset.prefetch(self.prefetch_batch_size)
        if num_take:
            self.last_test_filenames = self.test_filenames[:num_take]
            self.last_test_labels = self.test_labels[:num_take]
            return self.test_dataset.take(num_take)
        else:
            self.last_test_filenames = self.test_filenames
            self.last_test_labels = self.test_labels
            return self.test_dataset

    def batch_test_wrapper(self, estimator, input_fn:dict):
        result_dict = dict()
        for set_name, set_input_fn in input_fn.items():
            if set_input_fn is None:
                result_dict[set_name] = None
            else:
                sample_predictions = list(estimator.predict(input_fn=set_input_fn))
                sample_list = self.last_test_filenames
                sample_labels = self.last_test_labels

                result_dict[set_name] = (sample_list, sample_labels, sample_predictions)
        return result_dict
    
    def create_train_iterator(self):
        return self.train_dataset.make_one_shot_iterator().get_next()

    def create_test_iterator(self):
        return self.test_dataset.make_one_shot_iterator().get_next()

    def mat_file_checker(self, subjectID, filename, checklevel=None, supresswarn=False):
        pathobj = self.subject_path_mapper(subjectID)/filename
        if checklevel is None:
            checklevel = self.args.ignore_missing_mat_level
        if pathobj.exists():
            return True
        else:
            if checklevel < 2:
                raise IOError("Cannot find file {}".format(str(pathobj)))
            else:
                if not supresswarn:
                    print(">>>>WARNING: Cannot find file {}".format(str(pathobj)))
                return False
    
    def subject_path_mapper(self, subjectID) -> Path:
        """
        Map subjectID to the path where its data are stored.
        """
        subjectpath = Path(self.args.dataset_dir)/"Timecourses"/Path(str(subjectID))
        return subjectpath

    def subject_filter(self, subjectID_list, allow_partial:bool):
        mask = [True] * len(subjectID_list)
        for ind, subjectID in enumerate(subjectID_list):
            subjectpath = self.subject_path_mapper(subjectID)
            s_file_exist = {key: self.mat_file_checker(subjectID, filename, checklevel=100) 
                for key, filename in self.args.load_mat_dict.items()}
            if not any(s_file_exist.values()):
                mask[ind] = False
            elif not allow_partial and not all(s_file_exist.values()):
                mask[ind] = False
            if not mask[ind]:
                print("{} has been deleted.".format(subjectID))
        return mask
    
    def mat_file_loader(self, subjectID, filename):
        subjectpath = self.subject_path_mapper(subjectID)
        timeseries = scipy.io.loadmat(str(subjectpath/filename))['roiTC']
        return timeseries
    
cache = dict()

class corr_manager(data_manager):
    def gen_subject_tensor(self, subjectID):
        def correlation(time_series):
            """Compute the pairwise correlation between time series for ROIs."""
            corr = np.corrcoef(time_series, rowvar=False)
            corr = np.arctanh(corr-np.eye(corr.shape[0])) #pylint: disable=E1101
            corr_p = np.maximum(corr, 0) #pylint: disable=E1101
            corr_n = -np.minimum(corr, 0) #pylint: disable=E1101
            if self.args.column_normalize:
                corr_p = np.divide(corr_p, np.sum(corr_p, axis=0)) #pylint: disable=E1101
                corr_n = np.divide(corr_n, np.sum(corr_n, axis=0)) #pylint: disable=E1101
            return [corr_p, corr_n]
        
        load_mat_dict = self.args.load_mat_dict

        if subjectID in cache:
            corr_data = cache[subjectID]
        else:
            mat_data = {key: self.mat_file_loader(subjectID, filename) for key, filename in self.args.load_mat_dict.items()
                if self.mat_file_checker(subjectID, filename)}
            corr_data = {key: correlation(timeseries) for key, timeseries in mat_data.items()}
            cache[subjectID] = corr_data
        return list(chain(*corr_data.values()))

    def load_dataset(self, dataset):
        feature_length = len(self.args.load_mat_dict)
        tensor_gen_f = lambda subjectID: tf.py_func(self.gen_subject_tensor,
            [subjectID], [tf.float64, tf.float64]*feature_length, stateful=False)
        data_size = [264, 264]
        # print("Time series size: {}".format(data_size))
        # Set the shape of the tensor and then cast it to a supported type
        tensor_reshape_f = lambda tensors, label: \
            ({
                key: (
                    tf.cast(tf.reshape(next(tensors), data_size), tf.float32),
                    tf.cast(tf.reshape(next(tensors), data_size), tf.float32)
                    ) for key in self.args.load_mat_dict
                }, label)
        f = lambda subjectID, subjectlabel: tensor_reshape_f(iter(tensor_gen_f(subjectID)), subjectlabel)
        return dataset.map(f, self.args.num_parallel_calls)

class corr_mean_manager(data_manager):
    def gen_subject_tensor(self, subjectID):
        def correlation(time_series):
            """Compute the pairwise correlation between time series for ROIs."""
            corr = np.corrcoef(time_series, rowvar=False)
            corr = np.arctanh(corr-np.eye(corr.shape[0])) #pylint: disable=E1101
            return corr

        if subjectID in cache:
            corr_data = cache[subjectID]
        else:
            # Rest state data preprocessing
            mat_data = {key: self.mat_file_loader(subjectID, filename) for key, filename in self.args.load_mat_dict.items()
                if self.mat_file_checker(subjectID, filename)}
            corr_data = {key: correlation(timeseries) for key, timeseries in mat_data.items()}
            corr = np.mean(list(corr_data.values()), axis=0)
            corr_p = np.maximum(corr, 0) #pylint: disable=E1101
            corr_n = -np.minimum(corr, 0) #pylint: disable=E1101
            if self.args.column_normalize:
                corr_p = np.divide(corr_p, np.sum(corr_p, axis=0)) #pylint: disable=E1101
                corr_n = np.divide(corr_n, np.sum(corr_n, axis=0)) #pylint: disable=E1101
            corr_data = [corr_p, corr_n]
            cache[subjectID] = corr_data
        return corr_data

    def load_dataset(self, dataset):
        tensor_gen_f = lambda subjectID: tf.py_func(self.gen_subject_tensor,
            [subjectID], [tf.float64, tf.float64], stateful=False)
        data_size = [264, 264]
        # print("Time series size: {}".format(data_size))
        # Set the shape of the tensor and then cast it to a supported type
        tensor_reshape_f = lambda corr_p_norm, corr_n_norm, label: \
            ({self.args.selected_timeseries: (
                    tf.cast(tf.reshape(corr_p_norm, data_size), tf.float32),
                    tf.cast(tf.reshape(corr_n_norm, data_size), tf.float32)
                    )
                }, label)
        f = lambda subjectID, subjectlabel: tensor_reshape_f(*tensor_gen_f(subjectID), subjectlabel)
        return dataset.map(f, self.args.num_parallel_calls)
    