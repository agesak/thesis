import datetime
import argparse
import os
import re

from cod_prep.utils.misc import print_log_message
from cod_prep.claude.claude_io import makedirs_safely
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from thesis_utils.misc import str2bool, remove_if_output_exists
from thesis_utils.modeling import (read_in_data, create_train_test,
                                   random_forest_params,
                                   naive_bayes_params,
                                   svm_params, gbt_params,
                                   xgb_params,
                                   format_argparse_params)
from thesis_utils.model_evaluation import (get_best_fit,
                                           format_best_fit_params)


class ModelLauncher():

    model_dict = {"rf": "RandomForestClassifier",
                  "multi_nb": "MultinomialNB",
                  "bernoulli_nb": "BernoulliNB",
                  "complement_nb": "ComplementNB",
                  "svm": "SVC",
                  "svm_bag":"SVC",
                  "gbt": "GradientBoostingClassifier",
                  "xgb": "XGBClassifier"}
    param_dict = {"rf": 4,
                  "multi_nb": 1,
                  "bernoulli_nb": 1,
                  "complement_nb": 1,
                  "svm": 4,
                  "svm_bag":4,
                  "gbt": 4,
                  "xgb": 4}
    memory_dict = {"rf": 65,
                   "multi_nb": 8,
                   "bernoulli_nb": 6,
                   "complement_nb": 6,
                   "gbt": 30,
                   "xgb": 20,
                   "svm": 40,
                   "svm_bag":20}
    runtime_dict = {"rf": "52:00:00",
                    "multi_nb": "1:00:00",
                    "complement_nb": "1:00:00",
                    "bernoulli_nb": "1:00:00",
                    "gbt": "96:00:00",
                    "xgb": "14:00:00",
                    "svm": "96:00:00",
                    "svm_bag":"24:00:00"
                    }
    df_size_dict = {"x59": 1056994,
                    "y34": 1708834}
    num_datasets = 500
    # num_datasets = 10

    def __init__(self, run_filters):
        self.run_filters = run_filters
        self.test = self.run_filters["test"]
        self.phase = self.run_filters["phase"]
        self.int_cause = self.run_filters["int_cause"]
        self.model_types = self.run_filters["model_type"]
        self.code_system_id = self.run_filters["code_system_id"]
        if self.run_filters["description"] is None:
            self.description = "{:%Y_%m_%d}".format(datetime.datetime.now())
            if self.test:
                self.description += "_test"
            assert self.phase == "train_test", "your model run must be associated with an already existing train/test df date"
        else:
            self.description = self.run_filters["description"]
        self.model_dir = f"/ihme/cod/prep/mcod/process_data/{self.int_cause}/thesis/{self.description}"
        self.dataset_dir = f"/ihme/cod/prep/mcod/process_data/{self.int_cause}/thesis/sample_dirichlet/{self.description}"

    def create_training_data(self):
        makedirs_safely(self.model_dir)
        df = read_in_data(self.int_cause, self.code_system_id)
        train_df, test_df, int_cause_df = create_train_test(
            df, test=self.test, int_cause=self.int_cause)
        print_log_message("writing train/test to df")
        train_df.to_csv(f"{self.model_dir}/train_df.csv", index=False)
        test_df.to_csv(f"{self.model_dir}/test_df.csv", index=False)
        int_cause_df.to_csv(f"{self.model_dir}/int_cause_df.csv", index=False)

    def chunks(l, n):
        n = max(1, n)
        return (l[i:i + n] for i in range(0, len(l), n))

    def launch_create_testing_datasets(self):

        worker = f"/homes/agesak/thesis/analysis/create_test_datasets.py"
        makedirs_safely(self.dataset_dir)

        # if ModelLauncher.num_datasets == 10:
        #     numbers = (list(ModelLauncher.chunks(range(1, 11), 10)))
        if ModelLauncher.num_datasets == 500:
            numbers = (list(ModelLauncher.chunks(range(1, 501), 125)))
            dataset_dict = dict(zip(range(0, len(numbers)), numbers))
            holds_dict = {key: [] for key in dataset_dict.keys()}
            for batch in dataset_dict.keys():
                datasets = dataset_dict[batch]
                hold_ids = []
                for dataset_num in datasets:
                    params = [self.model_dir, self.dataset_dir, dataset_num,
                              ModelLauncher.df_size_dict[f"{self.int_cause}"]]
                    jobname = f"{self.int_cause}_dataset_{dataset_num}"
                    jid = submit_mcod(jobname, "python", worker,
                                      cores=2, memory="12G",
                                      params=params, verbose=True,
                                      logging=True, jdrive=False,
                                      queue="long.q", holds=holds_dict[batch])
                    hold_ids.append(jid)
                    if (dataset_num == datasets[-1]) & (batch != list(dataset_dict.keys())[-1]):
                        holds_dict.update({batch + 1: hold_ids})

    def launch_int_cause_predictions(self, short_name):
        predicted_test_dir = f"{self.dataset_dir}/{short_name}"

        params = [self.model_dir, predicted_test_dir, self.int_cause,
                  ModelLauncher.model_dict[short_name]]
        jobname = f"{ModelLauncher.model_dict[short_name]}_{self.int_cause}_predictions"
        worker = f"/homes/agesak/thesis/analysis/run_unobserved_predictions.py"
        submit_mcod(jobname, "python", worker, cores=2, memory="12G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="i.q")

    def launch_testing_models(self, model_name, short_name, best_model_params, dataset_num):

        best_model_dir = f"{self.model_dir}/{short_name}/model_{best_model_params}"
        testing_model_dir = f"{self.dataset_dir}/{short_name}"
        makedirs_safely(testing_model_dir)
        remove_if_output_exists(
            testing_model_dir, f"dataset_{dataset_num}_summary_stats.csv")
        remove_if_output_exists(
            testing_model_dir, f"dataset_{dataset_num}_predictions.csv")
        params = [best_model_dir, self.dataset_dir, testing_model_dir,
                  best_model_params, self.int_cause, dataset_num]
        jobname = f"{model_name}_{self.int_cause}_predictions_dataset_{dataset_num}_{best_model_params}"
        worker = f"/homes/agesak/thesis/analysis/run_testing_predictions.py"
        submit_mcod(jobname, "python", worker, cores=3, memory="20G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="i.q")

    def launch_training_models(self, model_name, short_name,
                               model_param):

        write_dir = f"{self.model_dir}/{short_name}/model_{model_param}"
        makedirs_safely(write_dir)
        # remove previous model runs
        remove_if_output_exists(write_dir, "grid_results.pkl")
        remove_if_output_exists(write_dir, "summary_stats.csv")

        params = [write_dir, self.model_dir, model_param,
                  model_name, short_name, self.int_cause]
        jobname = f"{short_name}_{self.int_cause}_{model_param}"
        worker = f"/homes/agesak/thesis/analysis/run_models.py"
        submit_mcod(jobname, "python", worker, cores=4,
                    memory=f"{ModelLauncher.memory_dict[short_name]}G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="long.q", runtime=f"{ModelLauncher.runtime_dict[short_name]}")

    def _launch_models(self, params, model_name, short_name):
        """helper function to launch training models"""

        for parameter in params:
            param = format_argparse_params(
                parameter, ModelLauncher.param_dict[short_name])
            self.launch_training_models(model_name, short_name,
                                        param)

    def launch(self):
        if self.phase == "train_test":
            self.create_training_data()

        if self.phase == "launch_training_model":
            for short_name in self.model_types:
                model_name = ModelLauncher.model_dict[short_name]
                if model_name == "RandomForestClassifier":
                    print_log_message("launching Random Forest")
                    params = random_forest_params(model_name)
                elif model_name == "MultinomialNB":
                    print_log_message("launching Multinomial Naive Bayes")
                    params = naive_bayes_params(model_name)
                elif model_name == "BernoulliNB":
                    print_log_message("launching Bernoulli Naive Bayes")
                    params = naive_bayes_params(model_name)
                elif model_name == "ComplementNB":
                    print_log_message("launching Complement Naive Bayes")
                    params = naive_bayes_params(model_name)
                elif model_name == "SVC":
                    print_log_message("launching SVC")
                    params = svm_params(model_name)
                elif model_name == "GradientBoostingClassifier":
                    params = gbt_params(model_name)
                elif model_name == "XGBClassifier":
                    params = xgb_params(model_name)
                print_log_message(
                    f"{len(params)} sets of model parameters")
                self._launch_models(params, model_name, short_name)

        if self.phase == "create_test_datasets":
            self.launch_create_testing_datasets()

        if self.phase == "launch_testing_models":
            for short_name in self.model_types:
                model_name = ModelLauncher.model_dict[short_name]
                # get parameters of best model fit for given model
                best_fit = get_best_fit(
                    model_dir=self.model_dir,
                    short_name=short_name)
                best_model_params = format_best_fit_params(
                    best_fit, model_name)
                num_datasets = len([x for i, x in enumerate(
                    os.listdir(self.dataset_dir)) if re.search(
                    "dataset_[0-9]{0,3}.csv", x)])
                failed = ModelLauncher.num_datasets - num_datasets
                assert num_datasets == ModelLauncher.num_datasets, f"{failed} jobs creating test datasets failed"
                for dataset in range(0, num_datasets):
                    dataset += 1
                    print(dataset)
                    self.launch_testing_models(
                        model_name, short_name, best_model_params, dataset)

        if self.phase == "launch_int_cause_predictions":
            for short_name in self.model_types:
                self.launch_int_cause_predictions(short_name=short_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--phase", help="", required=True,
        choices=["train_test", "launch_training_model",
                 "create_test_datasets", "launch_testing_models",
                 "launch_int_cause_predictions"])
    parser.add_argument("--test", type=str2bool, nargs="?",
                        const=True, default=False)
    parser.add_argument(
        "--int_cause", help="either x59 or y34", required=True,
        choices=["x59", "y34"])
    # not required
    parser.add_argument(
        "--model_type", help="short-hand name for ML classifier",
        choices=list(ModelLauncher.model_dict.keys()) + ["all"], nargs="*")
    parser.add_argument(
        "--code_system_id", help="1 is ICD 10, 6 is ICD9",
        choices=[1, 6], type=int)
    parser.add_argument(
        "--description",
        help='required for all phases except train_test',
        type=str
    )

    args = parser.parse_args()
    print(vars(args))
    launcher = ModelLauncher(vars(args))
    launcher.launch()
