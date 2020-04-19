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
        phase="format_map", source=["COL_DANE", "ZAF_STATSSA", "ITA_ISTAT"],
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

    if test:
        print_log_message(
            "THIS IS A TEST.. only using 5000 rows from each loc")
        df = df.merge(
            locs[["location_id", "parent_id", "level"]],
            on="location_id", how="left")
        # map subnationals to parent so
        # random sampling will be at country level
        df["location_id"] = np.where(
            df["level"] > 3, df["parent_id"],
            df["location_id"])
        df.drop(columns=["parent_id", "level"], inplace=True)
        # get a random sample from each location
        # bc full dataset takes forever to run
        dfs = []
        for loc in list(df.location_id.unique()):
            subdf = df.query(f"location_id=={loc}")
            random_df = subdf.sample(n=7000, replace=False)
            dfs.append(random_df)
        df = pd.concat(dfs, ignore_index=True, sort=True)

    # split train 75%, test 25%
    train_df, test_df = train_test_split(df, test_size=0.25)

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
    clf__estimator__max_features= df.loc[df[
        f"{model}"] == "clf__estimator__max_features",
        f"{model}_value"].str.split(",")[2]
    clf__estimator__criterion= df.loc[df[
        f"{model}"] == "clf__estimator__criterion",
        f"{model}_value"].str.split(",")[3]
    keys = "clf__estimator__n_estimators", "clf__estimator__max_depth", "clf__estimator__max_features", "clf__estimator__criterion"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        clf__estimator__n_estimators, clf__estimator__max_depth, clf__estimator__max_features, clf__estimator__criterion)]
    return params


def naive_bayes_params(model):
    assert (model == "MultinomialNB") | (model == "BernoulliNB") | (model == "ComplementNB"), "wrong model type"
    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    clf__estimator__alpha = df.loc[df[
        f"{model}"] == "clf__estimator__alpha",
        f"{model}_value"].str.split(",")[0]
    # well this is repetitive
    keys = "clf__estimator__alpha", "clf__estimator__alpha"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        clf__estimator__alpha)]
    return params

def svm_params(model):
    assert model == "SVC", "wrong model type"
    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    clf__estimator__C = df.loc[df[
        f"{model}"] == "clf__estimator__C",
        f"{model}_value"].str.split(",")[0]
    clf__estimator__kernel = df.loc[df[
        f"{model}"] == "clf__estimator__kernel",
        f"{model}_value"].str.split(",")[1]
    keys = "clf__estimator__C", "clf__estimator__kernel"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        clf__estimator__C, clf__estimator__kernel)]   
    return params

def gbt_params(model):
    assert model == "GradientBoostingClassifier"
    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    clf__estimator__n_estimators = df.loc[df[
        f"{model}"] == "clf__estimator__n_estimators",
        f"{model}_value"].str.split(",")[0]
    clf__estimator__learning_rate = df.loc[df[
        f"{model}"] == "clf__estimator__learning_rate",
        f"{model}_value"].str.split(",")[1]
    keys = "clf__estimator__n_estimators", "clf__estimator__learning_rate"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        clf__estimator__n_estimators, clf__estimator__learning_rate)]   
    return params

def format_argparse_params(param, param_len):
    assert len(param) == param_len, "error.. more than one set of params"
    # turn all params into a single "_" separated string for argparse
    param = "_".join([str(x) for x in list(param.values())])
    return param
