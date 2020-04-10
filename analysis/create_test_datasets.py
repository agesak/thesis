import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import sys

from cod_prep.utils.misc import print_log_message
from cod_prep.claude.claude_io import makedirs_safely
from thesis_utils.model_evaluation import generate_multiple_cause_rows
from analysis.launch_models import ModelLauncher


def _create_testing_datasets(cause, test_df, df_size, dirichlet_dict):

    # print_log_message(f"working on {cause}")
    df = pd.DataFrame({"cause_id": [np.NaN] * df_size})
    # proportion from dirichlet dictates how many rows are assigned to a given cause
    subdf = df.sample(frac=dirichlet_dict[cause], replace=False).assign(
        cause_id=cause)
    print_log_message(f"generating multiple cause rows for {cause}")
    subdf = generate_multiple_cause_rows(subdf, test_df, cause)

    return subdf


def create_test_datasets(test_df, metric_func, write_dir, dirichlet_dict,
                         causes, dataset_num, df_size):
    # multiprocessing here

    input_args = causes
    pool = Pool(12)
    _metric_func = partial(metric_func, test_df=test_df,
                           df_size=df_size, dirichlet_dict=dirichlet_dict)
    df_list = pool.map(_metric_func, input_args)

    pool.close()
    pool.join()

    df = pd.concat(df_list, sort=True, ignore_index=True)

    df_dir = f"{write_dir}/dataset_{dataset_num}"
    makedirs_safely(df_dir)
    print_log_message(f"writing dataset {dataset_num} to a df")
    df.to_csv(f"{df_dir}/dataset.csv", index=False)


# def create_test_datatsets(test_df, dirichlet_dict, write_dir, dataset_num, df_size):

#     df = pd.DataFrame({"cause_id": [np.NaN] * df_size})
#     dfs = []
#     for cause in dirichlet_dict.keys():
#         # proportion from dirichlet dictates how many rows are assigned to a given cause
#         subdf = df.sample(frac=dirichlet_dict[cause], replace=False).assign(
#             cause_id=cause)
#         print_log_message(f"generating multiple cause rows for {cause}")
#         ugh = generate_multiple_cause_rows(subdf, test_df, cause)
#         dfs.append(ugh)

#     df_dir = f"{write_dir}/dataset_{dataset_num}"
#     makedirs_safely(df_dir)

#     dfs = pd.concat(dfs, sort=True, ignore_index=True)
#     print_log_message(f"writing dataset {dataset_num} to a df")
#     dfs.to_csv(f"{df_dir}/dataset.csv", index=False)


def main(model_dir, write_dir, dataset_num, df_size):
    test_df = pd.read_csv(
        f"{model_dir}/test_df.csv")

    # dictionary of causes and their respective proportions in the data
    cause_distribution = test_df['cause_id'].value_counts(
        normalize=True).to_dict()
    # 500 dirichlet distributions based on test data cause distribution
    dts = np.random.dirichlet(alpha=list(
        cause_distribution.values()), size=1)

    # dictionary of cause ids to each dirichlet distribution
    dirichlet_dict = dict(zip(cause_distribution.keys(), dts[0]))
    causes = dirichlet_dict.keys()
    # also cause dirichlet_dict as object

    create_test_datasets(test_df, _create_testing_datasets,
                         write_dir, dirichlet_dict, causes,
                         dataset_num, df_size)
    # create_test_datatsets(test_df, dirichlet_dict, write_dir, dataset_num, df_size)


if __name__ == '__main__':

    # model_dir = str(sys.argv[1])
    # write_dir = str(sys.argv[2])
    # dataset_num = int(sys.argv[3])
    # df_size = int(sys.argv[4])
    model_dir = "/ihme/cod/prep/mcod/process_data/x59/thesis/2020_04_09"
    write_dir = "/ihme/cod/prep/mcod/process_data/x59/thesis/sample_dirichlet/rf/2020_04_09/200_60"
    dataset_num = 1
    df_size = 1000000

    main(model_dir, write_dir, dataset_num, df_size)
