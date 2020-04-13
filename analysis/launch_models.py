import pandas as pd
import datetime
import argparse
import os

from cod_prep.utils.misc import print_log_message
from cod_prep.claude.claude_io import makedirs_safely
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from thesis_utils.misc import str2bool
from thesis_utils.modeling import (read_in_data, create_train_test,
                                   random_forest_params,
                                   format_argparse_params)
from thesis_utils.model_evaluation import (get_best_fit,
                                           format_best_fit_params)


class ModelLauncher():

    model_dict = {"rf": "RandomForestClassifier",
                  "nb": "MultinomialNB",
                  "svm": "SVC",
                  "gbt": "GradientBoostingClassifier"}
    param_dict = {"rf": 2
                  }
    num_datasets = 100
    # df_size = 250000
    df_size = 1000000
    # but this doesnt work in the loop
    # model_dict.update("all":list(ModelLauncher.model_dict.values()))

    def __init__(self, run_filters):
        self.run_filters = run_filters
        self.test = self.run_filters["test"]
        self.phase = self.run_filters["phase"]
        self.int_cause = self.run_filters["int_cause"]
        self.model_types = self.run_filters["model_type"]
        if self.run_filters["description"] is None:
            self.description = "{:%Y_%m_%d}".format(datetime.datetime.now())
            if self.test:
                self.description += "_test"
            assert self.phase != "train_test", "your model run must be associated with an already existing train/test df date"
        else:
            self.description = self.run_filters["description"]
        self.model_dir = f"/ihme/cod/prep/mcod/process_data/{self.int_cause}/thesis"

    def create_training_data(self, model_dir):
        makedirs_safely(model_dir)
        df = read_in_data(self.int_cause)
        train_df, test_df = create_train_test(
            df, test=self.test, int_cause=self.int_cause)
        print_log_message("writing train/test to df")
        train_df.to_csv(f"{model_dir}/train_df.csv", index=False)
        test_df.to_csv(f"{model_dir}/test_df.csv", index=False)

    def compare_models(self, short_name, model_name, make_datasets):
        data_dir = f"{self.model_dir}/{self.description}"

        # get parameters of best model fit for given model
        best_fit = get_best_fit(
            model_dir=data_dir, short_name=short_name)
        best_model_params = format_best_fit_params(best_fit, model_name)
        print(best_model_params)

        # will use parameter specific folder for test datasets
        write_dir = f"{self.model_dir}/sample_dirichlet/{short_name}/{self.description}/{best_model_params}"
        makedirs_safely(write_dir)

        if make_datasets:
            for dataset_num in range(0, ModelLauncher.num_datasets):
                dataset_num += 1
                self.launch_create_testing_datasets(
                    write_dir, data_dir, dataset_num, best_model_params)

        return best_model_params, write_dir

    def launch_create_testing_datasets(self, write_dir, data_dir, dataset_num, best_model_params):

        params = [data_dir, write_dir, dataset_num, ModelLauncher.df_size]
        jobname = f"{self.int_cause}_dataset_{dataset_num}_{best_model_params}"
        worker = f"/homes/agesak/thesis/analysis/create_test_datasets.py"
        submit_mcod(jobname, "python", worker, cores=2, memory="12G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="i.q")

    def launch_testing_models(self, model_name, short_name, best_model_params,
                              write_dir, dataset_num):

        best_model_dir = f"{self.model_dir}/{self.description}/{short_name}/model_{best_model_params}"
        dataset_dir = f"{write_dir}/dataset_{dataset_num}"

        params = [best_model_dir, dataset_dir,
                  best_model_params, self.int_cause]
        jobname = f"{model_name}_{self.int_cause}_predictions_dataset_{dataset_num}_{best_model_params}"
        worker = f"/homes/agesak/thesis/analysis/run_predictions.py"
        submit_mcod(jobname, "python", worker, cores=2, memory="6G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="i.q")

    def launch_training_models(self, model_name, short_name,
                               model_param, model_dir):

        write_dir = f"{model_dir}/{short_name}/model_{model_param}"
        train_dir = f"{self.model_dir}/{self.description}"
        makedirs_safely(write_dir)

        params = [write_dir, train_dir, model_param,
                  model_name, short_name, self.int_cause]
        jobname = f"{model_name}_{self.int_cause}_{model_param}"
        worker = f"/homes/agesak/thesis/analysis/run_models.py"
        submit_mcod(jobname, "python", worker, cores=2, memory="6G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="i.q")

    def launch(self):
        if self.phase == "train_test":
            self.create_training_data(
                model_dir=f"{self.model_dir}/{self.description}")

        if self.phase == "launch_training_model":
            for short_name in self.model_types:
                model_name = ModelLauncher.model_dict[short_name]
                if model_name == "RandomForestClassifier":
                    print_log_message("launching Random Forest")
                    params = random_forest_params(model_name)
                    print_log_message(
                        f"{len(params)} sets of model parameters")
                    for parameter in params:
                        param = format_argparse_params(
                            parameter, ModelLauncher.param_dict[short_name])
                        self.launch_training_models(model_name, short_name,
                                                    param,
                                                    model_dir=f"{self.model_dir}/{self.description}")

        if self.phase == "create_test_datasets":
            for short_name in self.model_types:
                model_name = ModelLauncher.model_dict[short_name]
                best_model_params, write_dir = self.compare_models(
                    short_name, model_name, make_datasets=True)

        if self.phase == "launch_testing_models":
            for short_name in self.model_types:
                model_name = ModelLauncher.model_dict[short_name]
                best_model_params, write_dir = self.compare_models(
                    short_name, model_name, make_datasets=False)
                num_datasets = len(next(os.walk(write_dir))[1])
                failed = ModelLauncher.num_datasets-num_datasets
                assert num_datasets == ModelLauncher.num_datasets, f"{failed} jobs creating test datasets failed"
                for dataset in range(0, num_datasets):
                    dataset += 1
                    print(dataset)
                    self.launch_testing_models(
                        model_name, short_name, best_model_params,
                        write_dir, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--phase", help="", required=True,
        choices=["train_test", "launch_training_model",
                 "create_test_datasets", "launch_testing_models"])
    parser.add_argument("--test", type=str2bool, nargs="?",
                        const=True, default=False)
    parser.add_argument(
        "--int_cause", help="either x59 or y34", required=True,
        choices=["x59", "y34"])
    parser.add_argument(
        "--model_type", help="short-hand name for ML classifier",
        required=True,
        choices=list(ModelLauncher.model_dict.keys()) + ["all"], nargs="*")
    # not required
    parser.add_argument(
        "--description",
        help='only required for phase train_test',
        type=str
    )

    args = parser.parse_args()
    print(vars(args))
    launcher = ModelLauncher(vars(args))
    launcher.launch()
