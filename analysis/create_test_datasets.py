import pandas as pd
import numpy as np
import sys
from sklearn.externals import joblib

from cod_prep.utils.misc import print_log_message
from thesis_utils.model_evaluation import generate_multiple_cause_rows
from thesis_utils.misc import remove_if_output_exists, str2bool

DEM_COLS = ["cause_id", "location_id", "sex_id", "year_id", "age_group_id"]


def create_test_datatsets(test_df, dirichlet_dict, write_dir, dataset_num,
                          df_size, age_feature, dem_feature):

    df = pd.DataFrame({"cause_id": [np.NaN] * df_size})
    dfs = []
    for cause in dirichlet_dict.keys():
        print_log_message(cause)
        print_log_message(f"fraction is {dirichlet_dict[cause]}")
        # proportion from dirichlet dictates how many rows are assigned to a given cause
        subdf = df.sample(frac=dirichlet_dict[cause], replace=True).assign(
            cause_id=cause)
        print_log_message(f"generating multiple cause rows for {cause}")
        print_log_message(f"length of sample df is {len(subdf)}")
        mcause_df = generate_multiple_cause_rows(subdf, test_df, cause, age_feature, dem_feature)
        print_log_message(f"length of mcause df is {len(mcause_df)}")
        dfs.append(mcause_df)

    remove_if_output_exists(write_dir, f"dataset_{dataset_num}.csv")
    remove_if_output_exists(
        write_dir, f"dataset_{dataset_num}_dirichlet_distribution.pkl")

    print_log_message("about to concat")
    dfs = pd.concat(dfs, sort=True, ignore_index=True)
    print_log_message(f"length of df is {len(df)}")
    print_log_message(f"writing dataset {dataset_num} to a df")
    dfs.to_csv(f"{write_dir}/dataset_{dataset_num}.csv", index=False)

    joblib.dump(dirichlet_dict,
                f"{write_dir}/dataset_{dataset_num}_dirichlet_distribution.pkl")


def main(model_dir, write_dir, dataset_num, df_size, age_feature, dem_feature):
    test_df = pd.read_csv(
        f"{model_dir}/test_df.csv")

    # dictionary of causes and their respective proportions in the data
    cause_distribution = test_df['cause_id'].value_counts(
        normalize=True).to_dict()
    # 500 dirichlet distributions based on test data cause distribution
    alpha = list(cause_distribution.values())
    # multiply it by scalar so equals sum of uninformative alpha distribution (all 1's)
    alpha = [x*len(cause_distribution) for x in alpha]
    dts = np.random.dirichlet(alpha=alpha, size=1)
    print_log_message(dts)
    print_log_message(f"sum of dts is {int(dts.sum())}")

    # dictionary of cause ids to each dirichlet distribution
    dirichlet_dict = dict(zip(cause_distribution.keys(), dts[0]))
    print(dirichlet_dict)

    print_log_message("creating test datasets")
    create_test_datatsets(test_df, dirichlet_dict,
                          write_dir, dataset_num,
                          df_size, age_feature, dem_feature)


if __name__ == '__main__':

    model_dir = str(sys.argv[1])
    write_dir = str(sys.argv[2])
    dataset_num = int(sys.argv[3])
    df_size = int(sys.argv[4])
    age_feature = str2bool(sys.argv[5])
    dem_feature = str2bool(sys.argv[6])

    main(model_dir, write_dir, dataset_num, df_size, age_feature, dem_feature)

