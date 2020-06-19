import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from cod_prep.utils.misc import print_log_message
from thesis_utils.misc import remove_if_output_exists, str2bool
from thesis_utils.model_evaluation import generate_multiple_cause_rows

DEM_COLS = ["cause_id", "location_id", "sex_id", "year_id", "age_group_id"]


def create_test_datatsets(test_df, dirichlet_dict, write_dir, dataset_num,
                          df_size, age_feature, dem_feature):
    """Generate a test dataset of same length as the original test dataset
    Arguments:
        test_df: the actual test dataframe
        dirichlet_dict: dictionary mapping each cause id in actual test data
                        to its respective proportion in the test data
                        (generated from a Dirichlet distribution)
        write_dir: a directory to write each dataset to
        dataset_num: which dataset (of the 500) to create
        df_size: from the ModelLauncher, the desired size of the generated test
                 data (should be same size as the actual test data)
        age_feature: (Bool) - Do you want to include age as a feature?
        dem_feature: (Bool) - Do you want to include all demographic cols
                            (age, sex, year, and location) as features?
    """
    # create df of desired length
    df = pd.DataFrame({"cause_id": [np.NaN] * df_size})
    dfs = []
    # loop through each cause and generate rows with
    # multiple cause and demographic information
    for cause in dirichlet_dict.keys():
        # proportion from dirichlet dictates how many
        # rows are assigned to a given cause
        subdf = df.sample(frac=dirichlet_dict[cause], replace=True).assign(
            cause_id=cause)
        print_log_message(f"generating multiple cause rows for {cause}")
        mcause_df = generate_multiple_cause_rows(
            subdf, test_df, cause, age_feature, dem_feature)
        dfs.append(mcause_df)

    # if rerunning, remove previous dataset information
    remove_if_output_exists(write_dir, f"dataset_{dataset_num}.csv")
    remove_if_output_exists(
        write_dir, f"dataset_{dataset_num}_dirichlet_distribution.pkl")

    dfs = pd.concat(dfs, sort=True, ignore_index=True)
    print_log_message(f"writing dataset {dataset_num} to a df")
    # write generated test dataset to csv
    dfs.to_csv(f"{write_dir}/dataset_{dataset_num}.csv", index=False)
    # save randomly generated dirichlet distribution
    # in case need to exactly replicate
    joblib.dump(dirichlet_dict,
                f"{write_dir}/dataset_{dataset_num}_dirichlet_distribution.pkl"
                )


def main(model_dir, write_dir, dataset_num, df_size, age_feature, dem_feature):
    # read in actual test data
    test_df = pd.read_csv(
        f"{model_dir}/test_df.csv")

    # dictionary of causes and their respective proportions in the data
    cause_distribution = test_df['cause_id'].value_counts(
        normalize=True).to_dict()
    # alpha of dirichlet will be informed by cause distribution
    alpha = list(cause_distribution.values())
    # multiply it by scalar so equals sum of
    # uninformative alpha distribution (all 1's)
    # 4/21/2020 from Abie:  If you go with a more informative
    # alpha, you’ll need a way to justify the choice.
    # For example, you could keep the sum of alpha
    # the same as the uninformative.
    alpha = [x * len(cause_distribution) for x in alpha]
    dts = np.random.dirichlet(alpha=alpha, size=1)
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
