import pandas as pd
import glob
import os
import numpy as np
import itertools
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


def random_forest_params(model):
    assert model == "RandomForestClassifier", "wrong model type"

    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    clf__estimator__n_estimators = df.loc[df[
        f"{model}"] == "clf__estimator__n_estimators",
        f"{model}_value"].str.split(",")[0]

    # clf__estimator__max_features = df.loc[df[
    #     f"{model}"] == "clf__estimator__max_features",
    #     f"{model}_value"].tolist()
    clf__estimator__max_depth = df.loc[df[
        f"{model}"] == "clf__estimator__max_depth",
        f"{model}_value"].str.split(",")[1]
    keys = "clf__estimator__n_estimators", "clf__estimator__max_depth"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        clf__estimator__n_estimators, clf__estimator__max_depth)]

    return params


def format_argparse_params(param, param_len):
    # three bc 3 keys.. may change based on model? - could create model_type:param_number dictionary
    assert len(param) == param_len, "error.. more than one set of params"
    # turn all params into a single "_" separated string for argparse
    param = "_".join([str(x) for x in list(param.values())])
    return param
