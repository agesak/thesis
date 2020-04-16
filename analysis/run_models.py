import pandas as pd
import sys
from sklearn.externals import joblib

from cod_prep.utils.misc import print_log_message
from thesis_utils.grid_search import run_pipeline, format_gridsearch_params


def main(model_param, model_name, write_dir, train_dir, int_cause, short_name):

    model_df = pd.read_csv(f"{train_dir}/train_df.csv")
    print_log_message("formatting parameters")
    model_params = format_gridsearch_params(model_name, model_param)

    print_log_message("runninf pipeline")
    results, grid_results = run_pipeline(model_name, model_df, model_params,
                                         write_dir, int_cause)
    results.to_csv(f"{write_dir}/summary_stats.csv", index=False)
    joblib.dump(grid_results, f"{write_dir}/grid_results.pkl")


if __name__ == '__main__':

    write_dir = str(sys.argv[1])
    train_dir = str(sys.argv[2])
    model_param = str(sys.argv[3])
    model_name = str(sys.argv[4])
    short_name = str(sys.argv[5])
    int_cause = str(sys.argv[6])

    print(write_dir)
    print(train_dir)
    print(model_param)
    print(model_name)
    print(short_name)
    print(int_cause)
    main(model_param, model_name, write_dir, train_dir, int_cause, short_name)