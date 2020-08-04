import datetime
import argparse
import os
import re
from importlib import import_module

from cod_prep.utils.misc import print_log_message
from cod_prep.claude.claude_io import makedirs_safely
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from thesis_utils.misc import str2bool, remove_if_output_exists, chunks
from thesis_utils.modeling import (read_in_data, create_train_test,
                                   naive_bayes_params,
                                   format_argparse_params)
from thesis_utils.model_evaluation import get_best_fit


class ModelLauncher():

    model_dict = {"rf": "RandomForestClassifier",
                  "multi_nb": "MultinomialNB",
                  "bernoulli_nb": "BernoulliNB",
                  "complement_nb": "ComplementNB",
                  "svm": "SVC", "svm_bag": "SVC",
                  "gbt": "GradientBoostingClassifier",
                  "xgb": "XGBClassifier",
                  "nn": "KerasClassifier"}
    param_dict = {"rf": 4, "multi_nb": 1, "bernoulli_nb": 1,
                  "complement_nb": 1, "svm": 5, "svm_bag": 8, "gbt": 4,
                  "xgb": 5, "nn": 5}
    runtime_dict = {"rf": "140:00:00", "multi_nb": "1:00:00",
                    "complement_nb": "1:00:00", "bernoulli_nb": "1:00:00",
                    "gbt": "96:00:00", "xgb": "122:00:00",
                    "svm": "120:00:00", "svm_bag": "24:00:00",
                    "nn": "120:00:00"}
    df_size_dict = {"x59": 1056994, "y34": 1708834}
    num_datasets = 500
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
        self.dem_feature = self.run_filters["dem_feature"]
        self.icd_features = self.run_filters["icd_features"]
        self.most_detailed_locs = self.run_filters["most_detailed_locs"]
        if self.run_filters["description"] is None:
            description = "{:%Y_%m_%d}".format(datetime.datetime.now())
            if self.test:
                description += "_test"
            assert self.phase == "train_test", "your model run must be associated with an already existing train/test df date"
        else:
            description = self.run_filters["description"]
        self.description = description + "_" + self.icd_features
        self.dataset_dir = f"/ihme/cod/prep/mcod/process_data/{self.int_cause}/thesis/sample_dirichlet/{self.description}"
        self.model_dir = f"/ihme/cod/prep/mcod/process_data/{self.int_cause}/thesis/{self.description}"
        self.validate_run_filters()

    def validate_run_filters(self):
        assert not (self.age_feature &
                    self.by_age), "if you're running a model by age, it shouldn't also be a feature"
        assert not (self.age_feature &
                    self.dem_feature), "dem feature includes age, please choose just 1"
        assert ~self.by_age, "girl you decided not to run models by age for now"

    def create_training_data(self, df, age_group_id=None):
        if age_group_id:
            write_dir = f"{self.model_dir}/{age_group_id}"
        else:
            write_dir = f"{self.model_dir}"
        makedirs_safely(write_dir)
        train_df, test_df, int_cause_df = create_train_test(
            df, test=self.test, int_cause=self.int_cause, icd_feature=self.icd_features,
            age_group_id=age_group_id, most_detailed=self.most_detailed_locs)
        print_log_message(f"writing train/test to df for {age_group_id}")
        train_df.to_csv(f"{write_dir}/train_df.csv", index=False)
        test_df.to_csv(f"{write_dir}/test_df.csv", index=False)
        int_cause_df.to_csv(f"{write_dir}/int_cause_df.csv", index=False)

    def launch_create_testing_datasets(self, age_group_id=None):

        worker = f"/homes/agesak/thesis/analysis/create_test_datasets.py"
        if age_group_id:
            dataset_dir = f"{self.dataset_dir}/{age_group_id}"
            model_dir = f"{self.model_dir}/{age_group_id}"
        else:
            dataset_dir = self.dataset_dir
            model_dir = self.model_dir
        makedirs_safely(dataset_dir)

        numbers = (list(chunks(range(1, ModelLauncher.num_datasets+1), ModelLauncher.numbers)))
        dataset_dict = dict(zip(range(0, len(numbers)), numbers))
        holds_dict = {key: [] for key in dataset_dict.keys()}
        for batch in dataset_dict.keys():
            datasets = dataset_dict[batch]
            hold_ids = []
            for dataset_num in datasets:
                params = [model_dir, dataset_dir, dataset_num,
                          ModelLauncher.df_size_dict[f"{self.int_cause}"],
                          self.age_feature, self.dem_feature]
                jobname = f"{self.int_cause}_{self.icd_features}_dataset_{dataset_num}"
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
            jobname = f"{ModelLauncher.model_dict[short_name]}_{self.icd_features}_{self.int_cause}_predictions"

        params = [model_dir, predicted_test_dir, self.int_cause,
                  short_name, ModelLauncher.model_dict[short_name], self.age_feature, self.dem_feature]
        worker = f"/homes/agesak/thesis/analysis/run_unobserved_predictions.py"
        memory_dict = {"rf": 70, "multi_nb": 20, "bernoulli_nb": 20,
                       "complement_nb": 20, "xgb": 40, "svm": 40,
                       "svm_bag": 20, "nn": 350}
        submit_mcod(jobname, "python", worker, cores=2, memory=f"{memory_dict[short_name]}G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="long.q", runtime=ModelLauncher.runtime_dict[short_name])

    def launch_testing_models(self, model_name, short_name, best_model_params, age_group_id=None):

        if age_group_id:
            best_model_dir = f"{self.model_dir}/{age_group_id}/{short_name}/model_{best_model_params}"
            dataset_dir = f"{self.dataset_dir}/{age_group_id}"
            testing_model_dir = f"{dataset_dir}/{short_name}"
        else:
            best_model_dir = f"{self.model_dir}/{short_name}/model_{best_model_params}"
            testing_model_dir = f"{self.dataset_dir}/{short_name}"
            dataset_dir = self.dataset_dir

        makedirs_safely(testing_model_dir)
        worker = f"/homes/agesak/thesis/analysis/run_testing_predictions.py"
        memory_dict = {"rf": 120, "multi_nb": 30, "bernoulli_nb": 30,
                       "complement_nb": 30, "xgb": 40, "svm": 40,
                       "svm_bag": 20, "nn": 50}

        numbers = (list(chunks(range(1, ModelLauncher.num_datasets+1), int(ModelLauncher.num_datasets))))
        dataset_dict = dict(zip(range(0, len(numbers)), numbers))
        # to just launch a few (in one batch)
        # numbers = [29]
        # dataset_dict = {}
        # dataset_dict[0] = numbers
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
                          self.int_cause, dataset_num, self.age_feature,
                          self.dem_feature]
                jobname = f"{model_name}_{self.int_cause}_predictions_dataset_{dataset_num}_{best_model_params}_{self.icd_features}"
                memory = memory_dict[short_name]
                if (self.int_cause == "y34") & (short_name == "nn"):
                    memory = 150
                jid = submit_mcod(jobname, "python", worker,
                                  cores=4, memory=f"{memory}G",
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
            jobname = f"{short_name}_{self.icd_features}_{self.int_cause}_{model_param}"
            model_dir = self.model_dir

        memory_dict = {"rf": 150, "multi_nb": 20, "bernoulli_nb": 20,
                       "complement_nb": 20, "gbt": 30, "xgb": 40,
                       "svm": 40, "svm_bag": 20, "nn": 350}
        makedirs_safely(write_dir)
        # remove previous model runs
        remove_if_output_exists(write_dir, "grid_results.pkl")
        remove_if_output_exists(write_dir, "summary_stats.csv")

        params = [write_dir, model_dir, model_param,
                  model_name, short_name, self.int_cause,
                  self.age_feature, self.dem_feature]
        worker = f"/homes/agesak/thesis/analysis/run_models.py"
        memory = memory_dict[short_name]
        if (self.int_cause == "y34") & (short_name == "rf"):
            memory = "250"
        submit_mcod(jobname, "python", worker, cores=4,
                    memory=f"{memory}G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="long.q",
                    runtime=ModelLauncher.runtime_dict[short_name])

    def get_best_model(self, short_name, age_group_id):
        if age_group_id:
            model_dir = f"{self.model_dir}/{age_group_id}"
            dataset_dir = f"{self.dataset_dir}/{age_group_id}"
        else:
            model_dir = self.model_dir
            dataset_dir = self.dataset_dir

        best_model_params = get_best_fit(
            model_dir=model_dir, short_name=short_name)
        num_datasets = len([x for i, x in enumerate(
            os.listdir(dataset_dir)) if re.search(
            "dataset_[0-9]{0,3}.csv", x)])
        failed = ModelLauncher.num_datasets - num_datasets
        assert num_datasets == ModelLauncher.num_datasets, f"{failed} jobs creating test datasets failed"
        return best_model_params

    def check_datasets_exist(model_path):
        summaries = []
        for dataset_num in range(1, ModelLauncher.num_datasets + 1, 1):
            if not os.path.exists(f"{model_path}/dataset_{dataset_num}_summary_stats.csv"):
                summaries.append(dataset_num)
        assert len(summaries) == 0, f"datasets {summaries} failed"

    def launch(self):
        if self.phase == "train_test":
            df = read_in_data(int_cause = self.int_cause, code_system_id = self.code_system_id)
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
                        best_model_params = self.get_best_model(
                            short_name, age_group_id)
                        self.launch_testing_models(
                            model_name, short_name, best_model_params,
                            age_group_id)
                else:
                    best_model_params = self.get_best_model(
                        short_name, age_group_id=None)
                    self.launch_testing_models(
                        model_name, short_name, best_model_params,
                        age_group_id=None)

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
    parser.add_argument(
        "--int_cause", help="either x59 or y34", required=True,
        choices=["x59", "y34"])
    parser.add_argument("--icd_features", required=True,
                        choices=["grouped_ncode", "most_detailed", "aggregate_only",
                        "aggregate_and_letter", "most_detailed_and_letter"],
                        help="which ICD attributes to include as features in bow")
    parser.add_argument("--dem_feature", type=str2bool, nargs="?",
                        const=True, default=False, required=True,
                        help="include asyl as a feature in bow")
    parser.add_argument("--most_detailed_locs", type=str2bool, nargs="?",
                        const=True, default=False,
                        help="run model at most detailed location level")
    parser.add_argument("--age_feature", type=str2bool, nargs="?",
                        const=True, default=False,
                        help="include age as a feature in bow")
    parser.add_argument("--by_age", type=str2bool, nargs="?",
                        const=True, default=False,
                        help="run models by age")
    parser.add_argument("--test", type=str2bool, nargs="?",
                        const=True, default=False)
    # not required for train_test/create_test_datasets
    parser.add_argument(
        "--model_type", help="short-hand name for ML classifier",
        choices=list(ModelLauncher.model_dict.keys()), nargs="*")
    parser.add_argument(
        "--code_system_id", help="1 is ICD 10, 6 is ICD9",
        choices=[1, 6], type=int)
    parser.add_argument(
        "--description",
        help='just the date of the model run required for all phases except train_test',
        type=str
    )

    args = parser.parse_args()
    print(vars(args))
    launcher = ModelLauncher(vars(args))
    launcher.launch()
