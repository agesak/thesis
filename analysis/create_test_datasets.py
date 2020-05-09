import pandas as pd
import numpy as np
import sys
from sklearn.externals import joblib

from cod_prep.utils.misc import print_log_message
from thesis_utils.model_evaluation import generate_multiple_cause_rows
from thesis_utils.misc import remove_if_output_exists, str2bool

DEM_COLS = ["cause_id", "location_id", "sex_id", "year_id", "age_group_id"]


def create_test_datatsets(test_df, dirichlet_dict, write_dir, dataset_num,
                          df_size, age_feature):

    df = pd.DataFrame({"cause_id": [np.NaN] * df_size})
    dfs = []
    for cause in dirichlet_dict.keys():
        print_log_message(cause)
        # proportion from dirichlet dictates how many rows are assigned to a given cause
        subdf = df.sample(frac=dirichlet_dict[cause], replace=True).assign(
            cause_id=cause)
        print_log_message(f"generating multiple cause rows for {cause}")
        mcause_df = generate_multiple_cause_rows(subdf, test_df, cause, age_feature)
        dfs.append(mcause_df)

    # makedirs_safely(df_dir)
    remove_if_output_exists(write_dir, f"dataset_{dataset_num}.csv")
    remove_if_output_exists(
        write_dir, f"dataset_{dataset_num}_dirichlet_distribution.pkl")

    print_log_message("about to concat")
    dfs = pd.concat(dfs, sort=True, ignore_index=True)
    print_log_message(f"writing dataset {dataset_num} to a df")
    dfs.to_csv(f"{write_dir}/dataset_{dataset_num}.csv", index=False)

    joblib.dump(dirichlet_dict,
                f"{write_dir}/dataset_{dataset_num}_dirichlet_distribution.pkl")


def main(model_dir, write_dir, dataset_num, df_size, age_feature):
    test_df = pd.read_csv(
        f"{model_dir}/test_df.csv")

    # dictionary of causes and their respective proportions in the data
    cause_distribution = test_df['cause_id'].value_counts(
        normalize=True).to_dict()
    # 500 dirichlet distributions based on test data cause distribution
    dts = np.random.dirichlet(alpha=list(
        cause_distribution.values()), size=1)
    print_log_message(dts)

    # multiply it by scalar so equals sum of uninformative alpha distribution (all 1's)
    dts = dts * len(cause_distribution)
    print_log_message(dts)
    print_log_message(int(dts.sum()))

    # dictionary of cause ids to each dirichlet distribution
    dirichlet_dict = dict(zip(cause_distribution.keys(), dts[0]))
    print(dirichlet_dict)

    print_log_message("creating test datasets")
    create_test_datatsets(test_df, dirichlet_dict,
                          write_dir, dataset_num,
                          df_size, age_feature)


if __name__ == '__main__':

    model_dir = str(sys.argv[1])
    write_dir = str(sys.argv[2])
    dataset_num = int(sys.argv[3])
    df_size = int(sys.argv[4])
    age_feature = str2bool(sys.argv[5])
    main(model_dir, write_dir, dataset_num, df_size, age_feature)
