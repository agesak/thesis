import pandas as pd
import datetime
import itertools
import argparse

from cod_prep.utils.misc import print_log_message
from cod_prep.claude.claude_io import makedirs_safely
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from thesis_utils.modeling import read_in_data, create_train_test, str2bool


# function to read in all model outputs/precision metrics
# then put in table? or some easy way to compare them
# then compare to find best model parameters and save

class ModelLauncher():

    model_dict = {"rf": "RandomForestClassifier",
                  "nb": "MultinomialNB",
                  "svm": "SVC",
                  "gbt": "GradientBoostingClassifier"}
    # but this doesnt work in the loop
    # model_dict.update("all":list(ModelLauncher.model_dict.values()))

    def __init__(self, run_filters):
        self.run_filters = run_filters
        self.test = self.run_filters["test"]
        self.phase = self.run_filters["phase"]
        self.int_cause = self.run_filters["int_cause"]
        self.model_types = self.run_filters["model_type"]
        if self.run_filters["description"] == "":
            self.description = "{:%Y_%m_%d}".format(datetime.datetime.now())
            if self.test:
                self.description += "_test"
        else:
            self.description = self.run_filters["description"]
        self.model_dir = f"/ihme/cod/prep/mcod/process_data/x59/thesis/{self.description}"
        # print(self.test, type(self.test), self.phase, self.int_cause, self.description, self.model_dir)

    def random_forest_params(model):
        assert model == "RandomForestClassifier", "wrong model type"

        df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
        clf__estimator__n_estimators = df.loc[df[
            f"{model}"] == "clf__estimator__n_estimators",
            f"{model}_value"].str.split(",")[0]
        clf__estimator__max_features = df.loc[df[
            f"{model}"] == "clf__estimator__max_features",
            f"{model}_value"].tolist()
        clf__estimator__max_depth = df.loc[df[
            f"{model}"] == "clf__estimator__max_depth",
            f"{model}_value"].tolist()
        keys = "clf__estimator__n_estimators", "clf__estimator__max_features", "clf__estimator__max_depth"
        params = [dict(zip(keys, combo)) for combo in itertools.product(
            clf__estimator__n_estimators, clf__estimator__max_features,
            clf__estimator__max_depth)]

        return params

    def format_params(param):
        # three bc 3 keys.. may change based on model? - could create model_type:param_number dictionary
        assert len(param) == 3, "error.. more than one set of params"
        # turn all params into a single "_" separated string for argparse
        param = "_".join([str(x) for x in list(param.values())])
        return param

    def create_training_data(self):
        makedirs_safely(self.model_dir)
        df = read_in_data(self.int_cause)
        train_df, test_df = create_train_test(
            df, test=self.test, int_cause=self.int_cause)
        print_log_message("writing train/test to df")
        train_df.to_csv(f"{self.model_dir}/train_df.csv", index=False)
        test_df.to_csv(f"{self.model_dir}/test_df.csv", index=False)

    def launch_models(self, model, model_type, model_param):

        params = [model_param, model, self.model_dir, self.int_cause, model_type]
        jobname = f"{model}_{self.int_cause}_{model_param}"
        worker = f"/homes/agesak/thesis/analysis/run_models.py"
        submit_mcod(jobname, "python", worker, cores=2, memory="6G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="i.q")

    def launch(self):
        if self.phase == "train_test":
            self.create_training_data()

        if self.phase == "launch_model":
            for model_type in self.model_types:
                model = ModelLauncher.model_dict[model_type]
                if model == "RandomForestClassifier":
                    print_log_message("launching Random Forest")
                    params = ModelLauncher.random_forest_params(model)
                    print_log_message(
                        f"{len(params)} sets of model parameters")
                    for parameter in params:
                        param = ModelLauncher.format_params(parameter)
                        self.launch_models(model, model_type, param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--phase", help="", required=True,
        choices=["train_test", "launch_model"])
    parser.add_argument("--test", type=str2bool, nargs="?",
                        const=True, default=False)
    parser.add_argument(
        "--int_cause", help="", required=True,
        choices=["x59", "y34"])
    parser.add_argument(
        "--model_type", help="",
        required=True, choices=list(ModelLauncher.model_dict.keys()) + ["all"], nargs="*")
    # not required
    parser.add_argument(
        "--description", help='model run; if "launch_model" then date is appended',
        type=str
    )

    args = parser.parse_args()
    print(vars(args))
    launcher = ModelLauncher(vars(args))
    launcher.launch()
