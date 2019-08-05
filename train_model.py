import importlib, re, subprocess, sys
from foundations import arguments, run_training
from pathlib import Path

# Create argument parser
parser = arguments.create_argparser_tf("GroupINN")

# Load model defined in selected python module file under `models`
# Default file to load is `GroupINN.py`; it can be changed with --model_file option
model_file = parser.parse_known_args()[0].model_file #type: str
if model_file is None:
    GCN_module= importlib.import_module(".GroupINN", package="models")
else:
    model_file = re.sub(r"\.py$", "", model_file)
    print("===>Loading \"{}\" in module file \"{}\"".format("gcn_classification_net" ,model_file))
    GCN_module = importlib.import_module("."+model_file, package="models")

# Get model class
gcn_classification_net = getattr(GCN_module, "gcn_classification_net")

# Add model-specific argparse arguments 
gcn_classification_net.update_parser_argument(parser)

# Begin training process 
# and get the checkpoint path of the best-preforming model obtained in the training process
post_train_dict = run_training.train_classifier(parser, gcn_classification_net)

# Call checkpoint interpretation function
print("Analyzing checkpoint of best-preforming model {}".format(post_train_dict["best_checkpoint_path"]))
subprocess.run([sys.executable, "interpret_model.py", post_train_dict["best_checkpoint_path"]], cwd=Path(__file__).parent)
