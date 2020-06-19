import pandas as pd
import sys
from sklearn.externals import joblib

from thesis_utils.misc import str2bool
from cod_prep.utils.misc import print_log_message
from thesis_utils.grid_search import run_pipeline, format_gridsearch_params


def main(model_param, model_name, write_dir, train_dir,
         int_cause, short_name, age_feature, dem_feature):
    """Run gridsearch pipeline for a given classifier
    * Note this script is parallelized by parameter set
    (to allow for feasible run times)
    so each gridsearch object is fed only 1 set of model
    parameters, but this is done over a range of parameters
    Arguments:
        model_param: (str) - a single set of model parameters for
                     a given classifier
        model_name: the classifier name as defined by SciKit Learn
        write_dir: a directory to write the model object and summary to
        train_dir: a directory where the training dataset lives
        int_cause: the injuries garbage code of interest
        short_name: the abbreviated name for each classifier
                    defined in the ModelLauncher
        age_feature: (Bool) - Do you want to include age as a feature?
        dem_feature: (Bool) - Do you want to include all demographic cols
                            (age, sex, year, and location) as features?
    """
    # determine the model's feature vector
    if age_feature:
        x_col = "cause_age_info"
    elif dem_feature:
        x_col = "dem_info"
    else:
        x_col = "cause_info"

    print_log_message("reading in data")
    model_df = pd.read_csv(
        f"{train_dir}/train_df.csv")[["cause_id", f"{x_col}", f"{int_cause}"]]
    print_log_message("formatting parameters")
    model_params = format_gridsearch_params(short_name, model_param)

    print_log_message("running pipeline")
    results, grid_results = run_pipeline(model_name, short_name,
                                         model_df, model_params,
                                         write_dir, int_cause,
                                         age_feature, dem_feature)
    results.to_csv(f"{write_dir}/summary_stats.csv", index=False)
    joblib.dump(grid_results, f"{write_dir}/grid_results.pkl")


if __name__ == '__main__':

    write_dir = str(sys.argv[1])
    train_dir = str(sys.argv[2])
    model_param = str(sys.argv[3])
    model_name = str(sys.argv[4])
    short_name = str(sys.argv[5])
    int_cause = str(sys.argv[6])
    age_feature = str2bool(sys.argv[7])
    dem_feature = str2bool(sys.argv[8])

    main(model_param, model_name, write_dir,
         train_dir, int_cause, short_name,
         age_feature, dem_feature)
