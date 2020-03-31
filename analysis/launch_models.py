from datetime import datetime
import itertools
import argparse

from cod_prep.utils.misc import print_log_message
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from thesis_utils.modeling import read_in_data, create_train_test, str2bool


class ModelLauncher():

    def __init__(self, int_cause, run_filters):
        self.run_filters = run_filters
        self.test = self.run_filters["test"]
        self.phase = self.run_filters["phase"]
        self.int_cause = int_cause
        self.description = "{:%Y_%m_%d}".format(datetime.datetime.now())
        if self.test:
            self.description += "_test"
        self.model_dir = f"/ihme/cod/prep/mcod/process_data/x59/thesis/{self.description}"

    def random_forest_params():

        clf__estimator__n_estimators = [200, 500]
        clf__estimator__max_features = ["auto", "sqrt"]
        clf__estimator__max_depth = [1, 2] + [None]
        keys = "clf__estimator__n_estimators", "clf__estimator__max_features", "clf__estimator__max_depth"
        params = [dict(zip(keys, combo)) for combo in itertools.product(
            clf__estimator__n_estimators, clf__estimator__max_features,
            clf__estimator__max_depth)]

        return params

    def create_training_data(self):
        df = read_in_data(self.int_cause)
        train_df, test_df = create_train_test(df, test=self.test)
        print_log_message("writing train/test to df")
        train_df.to_csv(f"{self.model_dir}/train_df.csv", index=False)
        test_df.to_csv(f"{self.model_dir}/test_df.csv", index=False)

    def launch_models(self, model, model_param):

        params = [model_param, model, self.model_dir, self.int_cause]
        # idk how to differentiate names
        jobname = f"{model}_{self.int_cause}"
        worker = f"/homes/agesakk/thesis/analysis/run_models.py"
        submit_mcod(jobname, "python", worker, cores=2, memory="6G",
                    params=params, verbose=True, logging=True,
                    jdrive=False, queue="i.q")

    def launch(self):
        if self.phase == "train_test":
            create_train_test()

        if self.phase == "launch_model":
            for model in ["RandomForestClassifier"]:
                if model == "RandomForestClassifier":
                    params = ModelLauncher.random_forest_params()
                    for param in params:
                        self.launch_model(model, param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--phase", help="", required=True,
        choices=["train_test", "launch_model"])
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # this prbly doesnt work
    parser.add_argument("--test", type=str2bool,
                        const=True, default=False,
                        help="Activate nice mode.")

    args = parser.parse_args()
    print(vars(args))
    launcher = ModelLauncher(vars(args))
    launcher.launch()
