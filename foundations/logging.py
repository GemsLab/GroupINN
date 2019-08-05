from . import Path, os, tf, argparse, data_manager, np, pd, json
from collections import deque
import subprocess, scipy.stats, scipy.io

def calculate_F1(metrics: dict):
    f1_score = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    return f1_score

def calculate_pearson_correlation(labels, predictions, get_p_score=False):
    corr_coef, p_score = scipy.stats.pearsonr(labels, predictions)
    if not get_p_score:
        return corr_coef
    else:
        return corr_coef, p_score

def calculate_TPR_TNR(metrics: dict):
    result_dict = dict()

    result_dict["SP"] = metrics["TP"] + metrics["FN"]
    result_dict["TPR"] = metrics["TP"] / result_dict["SP"]

    result_dict["SN"] = metrics["TN"] + metrics["FP"]
    result_dict["TNR"] = metrics["TN"] / result_dict["SN"]

    return result_dict

def calculate_clsacc(metrics: dict):
    tpr_tnr_dict = calculate_TPR_TNR(metrics)
    return (tpr_tnr_dict["TPR"] + tpr_tnr_dict["TNR"]) / 2

def strip_var_group(metrics: dict):
    new_metrics = dict()
    for key, value in metrics.items():
        new_metrics[Path(key).name] = value
    return new_metrics

class prediction_writer:
    def __init__(self, csv_write_path=None):
        self.result_table = pd.DataFrame(columns=["sample_set", "true_label"])
        self.added_sample_set = []
        self.csv_write_path = csv_write_path

    def init_samples(self, sample_list, sample_label, sample_set=None):
        assert len(sample_list)==len(sample_label)
        for name, label in zip(sample_list, sample_label):
            self.result_table.loc[name] = np.NaN
            self.result_table.at[name, "sample_set"] = sample_set
            self.result_table.at[name, "true_label"] = label

    def record_result(self, epoch_name, sample_list, predicted_label, strict_match=True):
        assert len(sample_list) == len(predicted_label)
        if epoch_name not in self.result_table.columns:
            self.result_table[epoch_name] = np.NaN
        for name, label in zip(sample_list, predicted_label):
            if not strict_match and name not in self.result_table.index:
                self.result_table.loc[name] = np.NaN
            self.result_table.at[name, epoch_name] = label

    def print_result_table(self):
        print(self.result_table)

    def write_csv(self):
        print("Updating prediction results in {}...".format(self.csv_write_path), end="")
        self.result_table.to_csv(self.csv_write_path)
        print("Done!")

class prediction_writer_mat:
    def __init__(self, mat_write_path=None, pred_name="pred_var"):
        self.added_sample_set = []
        self.mat_write_path = Path(mat_write_path)
        self.mat_write_path.mkdir()
        self.sample_init_funcs = dict()
        self.result_dict = None
        self.pred_name = pred_name

    @staticmethod
    def make_init_sample(name, sample_set, label):
        def init_sample(sample_write_path):
            sample_path = sample_write_path / "{}_{}_{}.mat".format(name, sample_set, label)
            # with open(sample_path/"sample_info.json", "w") as info_out:
            #     json.dump({
            #         "sample_set": sample_set,
            #         "true_label": label
            #     }, info_out)
            return sample_path
        return init_sample

    def init_samples(self, sample_list, sample_label, sample_set=None):
        assert len(sample_list)==len(sample_label)
        for name, label in zip(sample_list, sample_label):
            self.sample_init_funcs[name] = self.make_init_sample(name, sample_set, label)

    def record_result(self, epoch_name, sample_list, predicted_var):
        assert len(sample_list) == len(predicted_var)
        epoch_path = self.mat_write_path / "epoch_{}".format(epoch_name)
        epoch_path.mkdir(exist_ok=True)
        self.result_dict = dict()
        for name, var in zip(sample_list, predicted_var):
            sample_path = self.sample_init_funcs[name](epoch_path)
            scipy.io.savemat(str(sample_path), {self.pred_name: var}, do_compression=True)
        self.write_csv = lambda: print("{} for each sample has been saved to {}".format(self.pred_name, epoch_path))

class evaluation_printer:
    def __init__(self, f1_avg_num=30):
        self.train_f1_queue = deque([], maxlen=f1_avg_num)
        self.test_f1_queue = deque([], maxlen=f1_avg_num)
        self.val_f1_queue = deque([], maxlen=f1_avg_num)

    def print_eval_result(self, train_eval_result=None,
            validation_eval_result=None, test_eval_result=None,
            update_f1_avg=True):

        print("{head:<13}\t{}".format(test_eval_result["global_step"], head="[global step]"))
        if train_eval_result:
            train_cf_dict = calculate_TPR_TNR(train_eval_result)
            train_f1 = calculate_F1(train_eval_result)
            print("{head:<13}\tloss: {loss:<11,.6f}{sep}accuracy: {accuracy:<6,.2%}{sep}TPR: {TPR:<6,.2%}({TP:>4,.0f}/{SP:<4,.0f}){sep}TNR: {TNR:.2%}({TN:>4,.0f}/{SN:<4,.0f})".format(
                    head="[Train set]", sep="    ", **train_eval_result, **train_cf_dict
                    ))
            print("{head:<13}\tprecision: {precision:<6,.2%}{sep}recall: {recall:<8,.2%}{sep}F1: {f1:<7,.2%} ({f1_avg_n} avg: {f1_avg:.2%})".format(
                    head="", f1=train_f1, f1_avg_n=len(self.train_f1_queue), f1_avg=np.mean(self.train_f1_queue), sep="    ", **train_cf_dict, **train_eval_result
            ))

        if validation_eval_result:
            val_cf_dict = calculate_TPR_TNR(validation_eval_result)
            val_f1 = calculate_F1(validation_eval_result)
            print("{head:<13}\tloss: {loss:<11,.6f}{sep}accuracy: {accuracy:<6,.2%}{sep}TPR: {TPR:<6,.2%}({TP:>4,.0f}/{SP:<4,.0f}){sep}TNR: {TNR:.2%}({TN:>4,.0f}/{SN:<4,.0f})".format(
                    head="[Valid set]", sep="    ", **validation_eval_result, **val_cf_dict
                    ))
            print("{head:<13}\tprecision: {precision:<6,.2%}{sep}recall: {recall:<8,.2%}{sep}F1: {f1:<7,.2%} ({f1_avg_n} avg: {f1_avg:.2%})".format(
                    head="", f1=val_f1, f1_avg_n=len(self.val_f1_queue), f1_avg=np.mean(self.val_f1_queue), sep="    ", **val_cf_dict, **validation_eval_result
            ))

        if test_eval_result:
            test_cf_dict = calculate_TPR_TNR(test_eval_result)
            test_f1 = calculate_F1(test_eval_result)
            print("{head:<13}\tloss: {loss:<11,.6f}{sep}accuracy: {accuracy:<6,.2%}{sep}TPR: {TPR:<6,.2%}({TP:>4,.0f}/{SP:<4,.0f}){sep}TNR: {TNR:.2%}({TN:>4,.0f}/{SN:<4,.0f})".format(
                    head="[Test set]", sep="    ", **test_eval_result, **test_cf_dict
            ))
            print("{head:<13}\tprecision: {precision:<6,.2%}{sep}recall: {recall:<8,.2%}{sep}F1: {f1:<7,.2%} ({f1_avg_n} avg: {f1_avg:.2%})".format(
                    head="", f1=test_f1, f1_avg_n=len(self.test_f1_queue), f1_avg=np.mean(self.test_f1_queue), sep="    ", **test_cf_dict, **test_eval_result
            ))

        if update_f1_avg and np.isfinite(train_f1) and np.isfinite(test_f1) and np.isfinite(val_f1): #pylint: disable=E1101
            if train_eval_result:
                self.train_f1_queue.append(train_f1)
            if validation_eval_result:
                self.val_f1_queue.append(val_f1)
            if test_eval_result:
                self.test_f1_queue.append(test_f1)

class Logging_Manager:
    def __init__(self, function_hooks:dict):
        self.function_hooks = function_hooks
        self.current_best = float("-inf")
        self.current_best_checkpoint_path = None
        self.previous = float("-inf")

    def add_parser_argument(self, parser:argparse.ArgumentParser):
        self.function_hooks["model_setup"].append(self.argparse_callback)
        group = parser.add_argument_group(title="Logging arguments")
        group.add_argument("--variables_export_frequency", default=0, type=int)
        group.add_argument("--variables_export_format",
            default="{epoch}-TrF{train_acc:.3f}-VaF{val_acc:.3f}-TeF{test_acc:.3f}-VaTP{TPR:.3f}-VaTN{TNR:.3f}")
        group.add_argument("--save_as_mat_prediction", default=[], nargs="+")

    def argparse_callback(self, args:argparse.Namespace):
        self.args = args
        self.export_dir = args.dir_format.format(r="export_dir", m=args.model_name, t=args.runname)
        Path(self.export_dir).mkdir(exist_ok=True, parents=True)
        self.writer = {
            "classes": prediction_writer(str(Path(self.export_dir)/"predicted_classes.csv"))
        }
        for pred_name in args.save_as_mat_prediction:
            self.writer[pred_name] = prediction_writer_mat(str(Path(self.export_dir)/pred_name), pred_name)
        self.printer = evaluation_printer()

        # Setup evaluation hooks
        self.function_hooks["eval"].append(self.print_eval_result)
        self.function_hooks["eval"].append(self.record_checkpoint)

        # Setup prediction hooks
        self.function_hooks["pred"].append(self.run_recorded_prediction)

        # Setup post_train hooks
        self.function_hooks["post_train"].append(self.get_best_model_checkpoint_path)

        # Save current arguments
        with open(Path(self.export_dir)/"arguments.json", "w") as arg_out:
            json.dump(args.raw_arg_dict, arg_out)

    def print_eval_result(self, epoch_name, eval_result_dict:dict):
        self.printer.print_eval_result(
            train_eval_result=eval_result_dict["train"],
            validation_eval_result=eval_result_dict["validation"],
            test_eval_result=eval_result_dict["test"]
        )

    def record_checkpoint(self, epoch_name, eval_result_dict:dict):
        args = self.args

        train_eval_result = eval_result_dict["train"]

        if eval_result_dict["validation"]:
            val_eval_result = eval_result_dict["validation"]
        else:
            val_eval_result = eval_result_dict["test"]
        test_eval_result = eval_result_dict["test"]

        export_abbr = ""
        if args.variables_export_frequency > 0 and (epoch_name % args.variables_export_frequency == 1 or epoch_name == args.train_epoch):
            export_abbr += "r"

        val_eval_result_f1 = calculate_F1(val_eval_result)
        val_eval_dict = calculate_TPR_TNR(val_eval_result)
        val_eval_tpr_tnr_diff = np.abs(val_eval_dict["TPR"] - val_eval_dict["TNR"])
        train_eval_clsacc = calculate_clsacc(train_eval_result)
        if (train_eval_clsacc >= 0.85) and (val_eval_tpr_tnr_diff <= 0.05) and (val_eval_result_f1 > self.current_best):
            export_abbr += "b"
            self.current_best = val_eval_result_f1

        if export_abbr != "":
            export_path = Path(self.export_dir)
            export_name = export_abbr + "_" + args.variables_export_format.format(
                epoch=epoch_name, train_acc=calculate_F1(train_eval_result), val_acc=calculate_F1(val_eval_result),
                test_acc=calculate_F1(test_eval_result), **calculate_TPR_TNR(val_eval_result)
            )
            export_path = export_path/export_name
            export_path.mkdir(parents=True, exist_ok=True)
            print("Copying checkpoint to path {}".format(str(export_path)))

            command = ["find", args.model_dir, "-maxdepth", "1", "-type", "f", "-name", "*",
                    "-exec", "cp", "--reflink=auto", "{}", str(export_path), ";"]
            subprocess.run(command)

        if "b" in export_abbr:
            global_step = train_eval_result["global_step"]
            self.current_best_checkpoint_path = str(Path(export_path) / "model.ckpt-{}".format(global_step))

    def run_recorded_prediction(self, epoch_name, pred_result_dict:dict):
        for set_name, prediction_result in pred_result_dict.items():
            if prediction_result is None:
                continue
            else:
                sample_list, sample_labels, predictions = prediction_result
            for key, writer in self.writer.items():
                if set_name not in writer.added_sample_set:
                    writer.init_samples(sample_list, sample_labels, sample_set=set_name)
                    writer.added_sample_set.append(set_name)
                predicted_label = [sample[key] for sample in predictions]
                writer.record_result(epoch_name, sample_list, predicted_label)

        for writer in self.writer.values():
            writer.write_csv()

    def get_best_model_checkpoint_path(self, post_train_dict:dict):
        post_train_dict["best_checkpoint_path"] = self.current_best_checkpoint_path

