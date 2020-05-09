import datetime
import argparse
import os
import re
from importlib import import_module

from cod_prep.utils.misc import print_log_message
from cod_prep.claude.claude_io import makedirs_safely
# from cod_prep.downloaders import create_age_bins
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from thesis_utils.misc import str2bool, remove_if_output_exists, chunks
from thesis_utils.modeling import (read_in_data, create_train_test,
                                   naive_bayes_params,
                                   format_argparse_params)
from thesis_utils.model_evaluation import (get_best_fit,
                                           format_best_fit_params)


class ModelLauncher():

    model_dict = {"rf": "RandomForestClassifier", "multi_nb": "MultinomialNB", "bernoulli_nb": "BernoulliNB", "complement_nb": "ComplementNB",
                  "svm": "SVC", "svm_bag": "SVC", "gbt": "GradientBoostingClassifier", "xgb": "XGBClassifier"}
    param_dict = {"rf": 4,
                  "multi_nb": 1,
                  "bernoulli_nb": 1,
                  "complement_nb": 1,
                  "svm": 5,
                  "svm_bag": 8,
                  "gbt": 4,
                  "xgb": 5}
    memory_dict = {"rf": 40,
                   "multi_nb": 8,
                   "bernoulli_nb": 20,
                   "complement_nb": 6,
                   "gbt": 30,
                   "xgb": 20,
                   "svm": 40,
                   "svm_bag": 20}
    runtime_dict = {"rf": "52:00:00",
                    "multi_nb": "1:00:00",
                    "complement_nb": "1:00:00",
                    "bernoulli_nb": "1:00:00",
                    "gbt": "96:00:00",
                    "xgb": "48:00:00",
                    "svm": "120:00:00",
                    "svm_bag": "24:00:00"
                    }
    df_size_dict = {"x59": 1056994,
                    "y34": 1708834}
    # num_datasets = 500
    num_datasets = 10
    # agg_ages = [39, 24, 224, 229, 47, 268, 294]
    agg_ages = [39]

    def __init__(self, run_filters):
        self.run_filters = run_filters
        self.test = self.run_filters["test"]
        self.phase = self.run_filters["phase"]
        self.int_cause = self.run_filters["int_cause"]
        self.model_types = self.run_filters["model_type"]
        self.code_system_id = self.run_filters["code_system_id"]
        self.age_feature = self.run_filters["age_feature"]
        self.by_age = self.run_filters["by_age"]
        assert ~(self.age_feature &
                 self.by_age), "if you're running a model by age, it shouldn't also be a feature"
        if self.run_filters["description"] is None:
            self.description = "{:%Y_%m_%d}".format(datetime.datetime.now())
            if self.test:
                self.description += "_test"
            assert self.phase == "train_test", "your model run must be associated with an already existing train/test df date"
        else:
            self.description = self.run_filters["description"]
        self.model_dir = f"/ihme/cod/prep/mcod/process_data/{self.int_cause}/thesis/{self.description}"
        self.dataset_dir = f"/ihme/cod/prep/mcod/process_data/{self.int_cause}/thesis/sample_dirichlet/{self.description}"

    def create_training_data(self, df, age_group_id=None):
        if age_group_id:
            write_dir = f"{self.model_dir}/{age_group_id}"
        else:
            write_dir = f"{self.model_dir}"
        makedirs_safely(write_dir)
        train_df, test_df, int_cause_df = create_train_test(
            df, test=self.test, int_cause=self.int_cause, age_group_id=age_group_id)
        print_log_message(f"writing train/test to df for {age_group_id}")
        train_df.to_csv(f"{write_dir}/train_df.csv", index=False)
        test_df.to_csv(f"{write_dir}/test_df.csv", index=False)
        int_cause_df.to_csv(f"{write_dir}/int_cause_df.csv", index=False)

    def launch_create_testing_datasets(self, age_group_id):

        worker = f"/homes/agesak/thesis/analysis/create_test_datasets.py"
        if age_group_id:
            dataset_dir = f"self.dataset_dir/{age_group_id}"
            model_dir = f"{self.model_dir}/{age_group_id}"            
        else:
            dataset_dir = self.dataset_dir
            model_dir = self.model_dir
        makedirs_safely(dataset_dir)

        if ModelLauncher.num_datasets == 10:
            numbers = (list(chunks(range(1, 11), 10)))
        # if ModelLauncher.num_datasets == 500:
        #     numbers = (list(chunks(range(1, 501), 250)))
            dataset_dict = dict(zip(range(0, len(numbers)), numbers))
            holds_dict = {key: [] for key in dataset_dict.keys()}
            for batch in dataset_dict.keys():
                datasets = dataset_dict[batch]
                hold_ids = []
                for dataset_num in datasets:
                    params = [model_dir, dataset_dir, dataset_num,
                              ModelLauncher.df_size_dict[f"{self.int_cause}"],
                              self.age_feature]
                    jobname = f"{self.int_cause}_dataset_{dataset_num}"
                    jid = submit_mcod(jobname, "python", worker,
                                      cores=2, memory="12G",
                                      params=params, verbose=True,
                                      logging=True, jdrive=False,
                                      queue="long.q", holds=holds_dict[batch])
                    hold_ids.append(jid)
                    if (dataset_num == datasets[-1]) & (batch != list(dataset_dict.keys())[-1]):
                        holds_dict.update({batch + 1: hold_ids})

    def launch_int_cause_predictions(self, short_name, age_group_id):
        if age_group_id:
            predicted_test_dir = f"{self.dataset_dir}/{age_group_id}/{short_name}"
            model_dir = f"{self.model_dir}/{age_group_id}"
            jobname = f"{ModelLauncher.model_dict[short_name]}_{self.int_cause}_{age_group_id}_predictions"
        else:
            predicted_test_dir = f"{self.dataset_dir}/{short_name}"
            model_dir = self.model_dir
            jobname = f"{ModelLauncher.model_dict[short_name]}_{self.int_cause}_predictions"

        params = [model_dir, predicted_test_dir, self.int_cause,
                  short_name, ModelLauncher.model_dict[short_name], self.age_feature]
        worker = f"/homes/agesak/thesis/analysis/run_unobserved_predictions.py"
        submit_mcod(jobname, "python", worker, cores=2, memory="12G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="i.q")

    def launch_testing_models(self, model_name, short_name, best_model_params, age_group_id=None):

        if age_group_id:
            best_model_dir = f"{self.model_dir}/{age_group_id}/{short_name}/model_{best_model_params}"
            dataset_dir = f"{self.dataset_dir}/{age_group_id}"
            testing_model_dir = f"{dataset_dir}/{short_name}"
            jobname = f"{model_name}_{self.int_cause}_predictions_dataset_{dataset_num}_{best_model_params}_{age_group_id}"
        else:
            best_model_dir = f"{self.model_dir}/{short_name}/model_{best_model_params}"
            testing_model_dir = f"{self.dataset_dir}/{short_name}"
            dataset_dir = self.dataset_dir
            jobname = f"{model_name}_{self.int_cause}_predictions_dataset_{dataset_num}_{best_model_params}"

        makedirs_safely(testing_model_dir)
        worker = f"/homes/agesak/thesis/analysis/run_testing_predictions.py"

        if ModelLauncher.num_datasets == 10:
            numbers = (list(chunks(range(1, 11), 5)))
        # if ModelLauncher.num_datasets == 500:
        #     numbers = (list(chunks(range(1, 501), 250)))
            dataset_dict = dict(zip(range(0, len(numbers)), numbers))
            holds_dict = {key: [] for key in dataset_dict.keys()}
            for batch in dataset_dict.keys():
                datasets = dataset_dict[batch]
                hold_ids = []
                for dataset_num in datasets:
                    remove_if_output_exists(
                        testing_model_dir,
                        f"dataset_{dataset_num}_summary_stats.csv")
                    remove_if_output_exists(
                        testing_model_dir,
                        f"dataset_{dataset_num}_predictions.csv")
                    params = [best_model_dir, dataset_dir,
                              testing_model_dir, best_model_params,
                              self.int_cause, dataset_num, self.age_feature]
                    jobname = f"{model_name}_{self.int_cause}_predictions_dataset_{dataset_num}_{best_model_params}"
                    jid = submit_mcod(jobname, "python", worker,
                                      cores=3, memory="25G",
                                      params=params, verbose=True,
                                      logging=True, jdrive=False,
                                      queue="long.q", holds=holds_dict[batch])
                    hold_ids.append(jid)
                    if (dataset_num == datasets[-1]) & (batch != list(dataset_dict.keys())[-1]):
                        holds_dict.update({batch + 1: hold_ids})

    def launch_training_models(self, model_name, short_name,
                               model_param, age_group_id=None):
        if age_group_id:
            write_dir = f"{self.model_dir}/{age_group_id}/{short_name}/model_{model_param}"
            jobname = f"{short_name}_{self.int_cause}_{model_param}_{age_group_id}"
            model_dir = f"{self.model_dir}/{age_group_id}"
        else:
            write_dir = f"{self.model_dir}/{short_name}/model_{model_param}"
            jobname = f"{short_name}_{self.int_cause}_{model_param}"
            model_dir = self.model_dir
        makedirs_safely(write_dir)
        # remove previous model runs
        remove_if_output_exists(write_dir, "grid_results.pkl")
        remove_if_output_exists(write_dir, "summary_stats.csv")

        params = [write_dir, model_dir, model_param,
                  model_name, short_name, self.int_cause,
                  self.age_feature]
        worker = f"/homes/agesak/thesis/analysis/run_models.py"
        submit_mcod(jobname, "python", worker, cores=4,
                    memory=f"{ModelLauncher.memory_dict[short_name]}G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="long.q",
                    runtime=f"{ModelLauncher.runtime_dict[short_name]}")

    def get_best_model(short_name, age_group_id):
        if age_group_id:
            model_dir = f"{self.model_dir}/{age_group_id}"
            dataset_dir = f"self.dataset_dir/{age_group_id}"
        else:
            model_dir = self.model_dir
            dataset_dir = self.dataset_dir

        best_fit = get_best_fit(model_dir=model_dir, short_name=short_name)
        best_model_params = format_best_fit_params(best_fit, short_name)
        num_datasets = len([x for i, x in enumerate(
            os.listdir(dataset_dir)) if re.search(
            "dataset_[0-9]{0,3}.csv", x)])
        failed = ModelLauncher.num_datasets - num_datasets
        assert num_datasets == ModelLauncher.num_datasets, f"{failed} jobs creating test datasets failed"
        return best_model_params

    def launch(self):
        if self.phase == "train_test":
            df = read_in_data(self.int_cause, self.code_system_id)
            if self.by_age:
                for age_group_id in ModelLauncher.agg_ages:
                    print_log_message(f"working on age: {age_group_id}")
                    self.create_training_data(df, age_group_id)
            else:
                self.create_training_data(df)

        if self.phase == "launch_training_model":
            for short_name in self.model_types:
                model_name = ModelLauncher.model_dict[short_name]
                if model_name in ["MultinomialNB", "BernoulliNB", "ComplementNB"]:
                    params = naive_bayes_params(short_name)
                else:
                    get_params = getattr(import_module(
                        f"thesis_utils.modeling"), f"{short_name}_params")
                    params = get_params(short_name)
                print_log_message(f"launching {model_name}")
                print_log_message(
                    f"{len(params)} sets of model parameters")
                for parameter in params:
                    param = format_argparse_params(
                        parameter, ModelLauncher.param_dict[short_name])
                    if self.by_age:
                        for age_group_id in ModelLauncher.agg_ages:
                            print_log_message(
                                f"launching models for age: {age_group_id}")
                            self.launch_training_models(
                                model_name, short_name, param, age_group_id)
                    else:
                        self.launch_training_models(
                            model_name, short_name, param)

        if self.phase == "create_test_datasets":
            if self.by_age:
                for age_group_id in ModelLauncher.agg_ages:
                    print_log_message(f"working on age: {age_group_id}")
                    self.launch_create_testing_datasets(age_group_id)
            else:
                self.launch_create_testing_datasets()

        if self.phase == "launch_testing_models":
            for short_name in self.model_types:
                model_name = ModelLauncher.model_dict[short_name]
                # get parameters of best model fit for given model
                if self.by_age:
                    for age_group_id in ModelLauncher.agg_ages:
                        best_model_params = get_best_model(
                            short_name, age_group_id)
                        self.launch_testing_models(
                            model_name, short_name, best_model_params, age_group_id)
                else:
                    best_model_params = get_best_model(
                        short_name, age_group_id=None)
                    self.launch_testing_models(
                        model_name, short_name, best_model_params, age_group_id=None)

        if self.phase == "launch_int_cause_predictions":
            for short_name in self.model_types:
                if self.by_age:
                    for age_group_id in ModelLauncher.agg_ages:
                        self.launch_int_cause_predictions(
                            short_name=short_name, age_group_id=age_group_id)
                else:
                    self.launch_int_cause_predictions(
                        short_name=short_name, age_group_id=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--phase", help="", required=True,
        choices=["train_test", "create_test_datasets",
                 "launch_training_model", "launch_testing_models",
                 "launch_int_cause_predictions"])
    parser.add_argument("--test", type=str2bool, nargs="?",
                        const=True, default=False)
    parser.add_argument(
        "--int_cause", help="either x59 or y34", required=True,
        choices=["x59", "y34"])
    parser.add_argument("--age_feature", type=str2bool, nargs="?",
                        const=True, default=False, required=True,
                        help="include age as a feature in bow")
    parser.add_argument("--by_age", type=str2bool, nargs="?",
                        const=True, default=False, required=True,
                        help="run models by age")
    # not required for train_test/create_test_datasets
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
