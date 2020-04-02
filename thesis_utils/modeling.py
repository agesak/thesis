import pandas as pd
import glob
import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

from mcod_prep.utils.mcause_io import get_mcause_data
from cod_prep.utils.misc import print_log_message
from db_queries import get_location_metadata
from thesis_utils.directories import get_limited_use_directory
from thesis_data_prep.launch_mcod_mapping import MCauseLauncher

BLOCK_RERUN = {"block_rerun": False, "force_rerun": True}


def read_in_data(int_cause):
    """Read in and append all MCoD data"""

    print_log_message("reading in not limited use data")
    # it"s not good the sources are hard-coded
    udf = get_mcause_data(
        phase="format_map", source=["COL_DANE", "ZAF_STATSSA"],
        sub_dirs=f"{int_cause}/thesis",
        data_type_id=9, assert_all_available=True,
        verbose=True, **BLOCK_RERUN)

    print_log_message("reading in limited use data")
    dfs = []
    for source in MCauseLauncher.limited_sources:
        limited_dir = get_limited_use_directory(source, int_cause)
        csvfiles = glob.glob(os.path.join(limited_dir, "*.csv"))
        for file in csvfiles:
            df = pd.read_csv(file)
            dfs.append(df)
    ldf = pd.concat(dfs, ignore_index=True, sort=True)
    df = pd.concat([udf, ldf], sort=True, ignore_index=True)

    return df


def create_train_test(df, test, int_cause):
    """Create train/test datasets, if running tests,
    randomly sample from all locations so models don't take forever to run"""

    locs = get_location_metadata(gbd_round_id=6, location_set_id=35)

    # split train 75%, test 25%
    train_df, test_df = train_test_split(df, test_size=0.25)

    # will need to do this for test df too..
    if test:
        print_log_message(
            "THIS IS A TEST.. only using 5000 rows from each loc")
        train_df = train_df.merge(
            locs[["location_id", "parent_id", "level"]],
            on="location_id", how="left")
        # map subnationals to parent so
        # random sampling will be at country level
        train_df["location_id"] = np.where(
            train_df["level"] > 3, train_df["parent_id"],
            train_df["location_id"])
        train_df.drop(columns=["parent_id", "level"], inplace=True)
        # get a random sample from each location
        # bc full dataset takes forever to run
        dfs = []
        for loc in list(train_df.location_id.unique()):
            subdf = train_df.query(f"location_id=={loc}")
            random_df = subdf.sample(n=5000, replace=False)
            dfs.append(random_df)
        train_df = pd.concat(dfs, ignore_index=True, sort=True)

    # will only train/test where we know UCoD
    # see how final results change when subsetting to where x59==0 -
    # so basically filtering out rows where
    # x59 in chain but ucod is gbd injuries cause
    train_df = train_df[["cause_id", "cause_info",
                         f"{int_cause}"]].query("cause_id!=743")

    return train_df, test_df


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# def measure_prediction_quality(csmf_pred, y_test):
#     """Calculate population-level prediction quality (CSMF Accuracy)
    
#     Parameters
#     ----------
#     csmf_pred : pd.Series, predicted distribution of causes
#     y_test : array-like, labels for test dataset
    
#     Results
#     -------
#     csmf_acc : float
#     """
    
#     csmf_true = pd.Series(y_test).value_counts() / float(len(y_test))
#     temp = np.abs(csmf_true-csmf_pred)
#     csmf_acc = 1 - temp.sum()/(2*(1-np.min(csmf_true)))
#     return csmf_acc


def calculate_cccsmfa(y_true, y_pred):
    # https://stackoverflow.com/questions/32401493/how-to-create-customize-your-own-scorer-function-in-scikit-learn
    # built from https://github.com/aflaxman/siaman16-va-minitutorial/blob/master/1-tutorial-notebooks/4-va_csmf.ipynb
    # y true is true cause ids
    # y pred is predicted cause ids

    random_allocation = 0.632
    
    csmf_true = pd.Series(y_true).value_counts()/float(len(y_true))
    csmf_pred = pd.Series(y_pred).value_counts()/float(len(y_pred))
    numerator = np.abs(csmf_true - csmf_pred)
    # first get csmfa
    csmfa = (1 - numerator.sum())/(2*(1-np.min(csmf_true)))

    # then get cccsmfa
    cccsmfa = (csmfa-random_allocation)/(1-random_allocation)

    return cccsmfa

