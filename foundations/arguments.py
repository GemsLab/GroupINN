from . import Path, argparse, tf, datetime, os, tf_utils, deque, data_manager, logging, json
from abc import ABC, abstractmethod
import copy, re
from io import StringIO

config_manager_dict = dict()
function_hooks = {
    "argparse": [], "model_setup": [], "eval": [], "pred": [], "post_train": []
}

def add_loss_weights_argument(parser, cls, model_name:str):
    def _get_weights_dict(cls):
        weights_dict = {name: getattr(cls, name) for name in dir(cls) if not name.startswith("_")}
        return weights_dict

    def _print_current_weights(cls):
        weights_dict = cls._get_weights_dict()
        print("Current loss weights: {}".format(weights_dict))

    def _update_parser_argument(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="{} loss weights".format(model_name))
        weights_dict = cls._get_weights_dict()
        for name, value in weights_dict.items():
            group.add_argument("--{}".format(name), type=float, default=value,
                                help="(default: {})".format(value))
        args, _ = parser.parse_known_args()
        args_dict = vars(args)
        for name in weights_dict:
            if name in args_dict:
                setattr(cls, name, args_dict[name])
        cls._print_current_weights()
        return parser
    
    def _argparse_callback(cls, args:argparse.Namespace):
        weights_dict = cls._get_weights_dict()
        for key in weights_dict:
            setattr(cls, key, getattr(args, key))
    
    setattr(cls, "_get_weights_dict", classmethod(_get_weights_dict))
    setattr(cls, "_print_current_weights", classmethod(_print_current_weights))
    setattr(cls, "_update_parser_argument", classmethod(_update_parser_argument))
    setattr(cls, "_argparse_callback", classmethod(_argparse_callback))
    cls._update_parser_argument(parser)
    function_hooks["argparse"].append(cls._argparse_callback)

class Config_Manager(ABC):
    '''
    Base class for argument manager
    '''
    def __init__(self, function_hooks:dict):
        self.function_hooks = function_hooks
    
    @abstractmethod
    def add_parser_argument(self, parser:argparse.ArgumentParser=None):
        self.function_hooks["argparse"].append(self.argparse_callback)

    def argparse_callback(self, args: argparse.Namespace):
        '''
        This callback function will parse and process the input arguments from command line
        See `parse_args` function in this module for details
        '''
        pass

class Basic_Config_Manager(Config_Manager):
    def __init__(self, function_hooks, model_name:str):
        super().__init__(function_hooks)
        self.starttime = datetime.now().strftime("%Y%m%d/%H%M%S")
        self.model_name = model_name
        self.data_manager = None
    
    def add_parser_argument(self, parser: argparse.ArgumentParser):
        super().add_parser_argument()
        parser.add_argument("--config_file", "-c", default=None, nargs="+", help="Path to a json config file.")
        parser.add_argument("--model_file", default=None, help="Path to the model definition.")
        parser.add_argument("--model_name", default=self.model_name, help="(default: %(default)s)")
        parser.add_argument("--dir_format",
            default=str(Path(__file__).parents[1]/"{r}"/"{m}"/"{t}"))

    def argparse_callback(self, args: argparse.Namespace):
        args.runname = self.starttime
        if args.message is not None:
            args.runname = args.runname + "-" + args.message

        if args.model_file is not None:
            args.model_name = re.sub(r'\[[^\[\]]*\]', re.sub(r"\.py$", "", args.model_file), args.model_name)
        else:
            args.model_name = re.sub(r'\[|\]', '', args.model_name)
        self.model_name = args.model_name
        
        self.args = args
        args.raw_arg_dict = vars(copy.deepcopy(args))

    def set_model_dir(self):
        args = self.args
        args.model_dir = args.dir_format.format(r="model_dir", m=args.model_name, t=args.runname)
        print("===> Model dir set to {}".format(args.model_dir))

def add_rt_arguments(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_rt", dest="use_rt", action="store_true")
    group.add_argument("--no_use-rt", dest="use_rt", action="store_false")
    group.set_defaults(use_rt=True)

###
class Data_Config_Manager(Config_Manager):
    avaliable_timeseries = {
        "emotionLR": "power_264_tfMRI_EMOTION_LR.mat", "emotionRL": "power_264_tfMRI_EMOTION_RL.mat",
        "gamblingLR": "power_264_tfMRI_GAMBLING_LR.mat" , "gamblingRL": "power_264_tfMRI_GAMBLING_RL.mat",
        "socialLR": "power_264_tfMRI_SOCIAL_LR.mat", "socialRL": "power_264_tfMRI_SOCIAL_RL.mat",
        "wmLR": "power_264_tfMRI_WM_LR.mat", "wmRL": "power_264_tfMRI_WM_RL.mat"
    }
    
    def __init__(self, function_hooks:dict):
        assert len(self.avaliable_timeseries.values()) == len(set(self.avaliable_timeseries.values()))
        super().__init__(function_hooks)

    def add_parser_argument(self, parser: argparse.ArgumentParser):
        super().add_parser_argument()
        group = parser.add_argument_group(title="task control arguments")
        group.add_argument("--dataset_dir", default="dataset/HCP", help="(default: %(default)s)")
        group.add_argument("--meta_filename", default="dataset/HCP/966_phenotypics.csv", help="(default: %(default)s)")
        group.add_argument("--selected_timeseries", nargs="+", default=["wmMEAN"],
            help=self.show_avaliable_timeseries() + "\n" +
                "Select timeseries that you would like to feed into your network.\n"
                "When input argument ends with \"MEAN\", the timeseries whose name beginning with\n" 
                "the same word as it is presented before \"MEAN\" will be loaded\n"
                "and their average would be the input.\n" 
                "Like \"wmMEAN\" average all timeseries with name beginning with \"wm\"\n"
                "(default: %(default)s)"
            )
        group.add_argument("--validation_proportion", default=0.1, type=float)
        group.add_argument("--validation_selection_seed", default=None, type=int)
        group.add_argument("--training_shuffle_seed", default=None, type=int)
        group.add_argument("--ignore_missing_mat_level", default=1, type=int,
            help=("0 for strict mode: all subjects in training/testing set must have corresponding files;\n"
                "1 for semi-strict mode: a subject will be skipped when any corresponding files is missing;\n"
                "2 for flexible mode: a subject will be skipped only if all his correspoding input files are missing.\n"
                    "(default: %(default)s)"))
    
    def argparse_callback(self, args: argparse.Namespace):
        args.selected_timeseries, data_manager_name, args.load_mat_dict = self.parse_selected_timeseries(args.selected_timeseries)
        manager = getattr(data_manager, data_manager_name)(args)
        config_manager_dict["Basic"].data_manager = manager
        
    def parse_selected_timeseries(self, selected_timeseries:list):
        assert type(selected_timeseries) is list
        new_timeseries_name = None
        needed_timeseries = dict()
        selected_data_manager = None
        if len(selected_timeseries) != 1:
            if "MEAN" in selected_timeseries:
                selected_data_manager = "corr_mean_manager"
                need_iter = [key for key in selected_timeseries if key != "MEAN"]
                new_timeseries_name = "-".join(need_iter)+"-MEAN"
                for key in need_iter:
                    needed_timeseries[key] = self.avaliable_timeseries[key]
            else:
                raise ValueError(
                    "Model does not accept multiple correlation matrices as input.\n"
                    "Add 'MEAN' to the --selected_timeseries argument list so that multiple correlation matrices will be averaged before sent into the model."
                    )
        
        if len(selected_timeseries) == 1:
            selected_timeseries = selected_timeseries[0]
            if selected_timeseries.endswith("MEAN"):
                selected_data_manager = "corr_mean_manager"
                new_timeseries_name = selected_timeseries
                search_abbr = selected_timeseries.replace("MEAN", "")
                if search_abbr == "task":
                    need_iter = [key for key in self.avaliable_timeseries.keys() if not key.startswith("rest")]
                else:
                    need_iter = [key for key in self.avaliable_timeseries.keys() if key.startswith(search_abbr)]
                for key in need_iter:
                    needed_timeseries[key] = self.avaliable_timeseries[key]
            else:
                selected_data_manager = "corr_manager"
                new_timeseries_name = selected_timeseries
                needed_timeseries[selected_timeseries] = self.avaliable_timeseries[selected_timeseries]
        print("===> Following timeseries files will be loaded: {}".format(needed_timeseries))
        print("===> Selected data manager: {}".format(selected_data_manager))
        return new_timeseries_name, selected_data_manager, needed_timeseries

    def show_avaliable_timeseries(self, item_per_line = 6):
        text = StringIO()
        num_key = len(self.avaliable_timeseries)
        len_key_max = max([len(key) for key in self.avaliable_timeseries.keys()])
        text.write("================avaliable timeseries================\n")
        for ind, key in enumerate(self.avaliable_timeseries.keys()):
            if (ind + 1) % item_per_line == 0 and (ind + 1) != num_key:
                text.write("{:<{width}}\n".format(key, width=len_key_max+2))
            else:
                text.write("{:<{width}}".format(key, width=len_key_max+2))
        text.write("\n====================================================")
        return text.getvalue()

class Train_Config_Manager(Config_Manager):
    def add_parser_argument(self, parser: argparse.ArgumentParser):
        super().add_parser_argument()
        group = parser.add_argument_group(title="train control arguments")
        group.add_argument("--restore_dir", default=None)
        group.add_argument("--batch_size", default=16, type=int)
        group.add_argument("--train_epoch", default=300, type=int)
        group.add_argument("--column_normalize", default=False, action="store_true")
        group.add_argument("--message", "-m", default=None, help="Comments appended after runname")
    
    def argparse_callback(self, args: argparse.Namespace):
        tf_utils.handle_restore_dir(args)

class TensorFlow_Config_Manager(Config_Manager):
    def add_parser_argument(self, parser: argparse.ArgumentParser):
        super().add_parser_argument()
        group = parser.add_argument_group(title="TensorFlow arguments")
        group.add_argument("--largeGPU", default=False, action="store_true")
        group.add_argument("--verbosity", "-v", default=tf.logging.WARN, type=int)
        group.add_argument("--num_parallel_calls", default=4, type=int, help="(default: %(default)s)")
        group.add_argument("--tf_cpp_verbosity", default="1", type=str,
            help=("0 for all logs shown;\n1 to filter out INFO logs;\n"
                  "2 to additionally filter out WARNING logs;\n"
                  "3 to additionally filter out ERROR logs.\n"
                  "(default: %(default)s)"))
        group.add_argument("--tf_random_seed", default=None, type=int)
        group.add_argument("--no_tensorboard", default=False, action="store_true")
    
    def argparse_callback(self, args: argparse.Namespace):
        tf.logging.set_verbosity(args.verbosity)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = args.tf_cpp_verbosity
        session_config = tf.ConfigProto()
        if args.largeGPU:
            session_config.gpu_options.per_process_gpu_memory_fraction = 0.45 #pylint: disable=E1101
        self.estimator_config = tf.estimator.RunConfig(session_config=session_config)
        if args.tf_random_seed:
            tf.set_random_seed(args.tf_random_seed)
    

def create_argparser_tf(model_name:str):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, add_help=False)
    config_manager_dict["Basic"] = Basic_Config_Manager(function_hooks, model_name)
    config_manager_dict["Data"] = Data_Config_Manager(function_hooks)
    config_manager_dict["Train"] = Train_Config_Manager(function_hooks)
    config_manager_dict["Logging"] = logging.Logging_Manager(function_hooks)
    config_manager_dict["TensorFlow"] = TensorFlow_Config_Manager(function_hooks)

    config_manager_dict["Basic"].add_parser_argument(parser)
    add_rt_arguments(parser)
    config_manager_dict["Data"].add_parser_argument(parser)
    config_manager_dict["Train"].add_parser_argument(parser)
    config_manager_dict["Logging"].add_parser_argument(parser)
    config_manager_dict["TensorFlow"].add_parser_argument(parser)

    return parser

def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    args = parser.parse_args()

    if args.config_file:
        for config_file in args.config_file:
            config_path = Path(config_file)
            
            if not config_path.exists():
                parent_path = Path("configs")
                config_path = parent_path / config_path.with_suffix(".json")
            
            print("===> Loading config file {}".format(str(config_path)))
            with open(str(config_path), "r") as f:
                config_dict = json.load(f) # type: dict
            for key, value in config_dict.items():
                if hasattr(args, key):
                    print("Setting \"{}\" to \"{}\"".format(key, value))
                    if key == "message":
                        args.message += "_{}".format(value)
                    else:
                        setattr(args, key, value)
                else:
                    raise KeyError("Key {} is not specified in the argparser.".format(key))

    for hook_func in function_hooks["argparse"]:
        hook_func(args)
    
    return args