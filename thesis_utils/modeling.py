import pandas as pd
import glob
import os
import numpy as np
import itertools
from sklearn.model_selection import train_test_split

from mcod_prep.utils.mcause_io import get_mcause_data
from mcod_prep.utils.nids import get_datasets
from cod_prep.utils.misc import print_log_message
from cod_prep.downloaders import (get_ages, add_age_metadata, create_age_bins,
    add_cause_metadata, get_current_cause_hierarchy)
from cod_prep.claude.configurator import Configurator
from db_queries import get_location_metadata
from thesis_utils.directories import get_limited_use_directory
from thesis_data_prep.launch_mcod_mapping import MCauseLauncher

AGG_AGES = [39, 24, 224, 229, 47, 268, 294]
BLOCK_RERUN = {"block_rerun": False, "force_rerun": True}
DEM_COLS = ["cause_id", "location_id", "sex_id", "year_id", "age_group_id"]
CONF = Configurator('standard')


def read_in_data(int_cause, inj_garbage=False, code_system_id=None):
    """Read in and append all MCoD data"""
    # col, zaf, and ita dont have icd 9
    print_log_message("reading in not limited use data")
    if inj_garbage:
        print_log_message(
            "writing formatted df with only nonX59/Y34 garbage codes as UCOD")
        subdirs = f"{int_cause}/thesis/inj_garbage"
    else:
        subdirs = f"{int_cause}/thesis"
    # it"s not good the sources are hard-coded
    if code_system_id != 6:
        # col, zaf, and ita dont have icd 9
        udf = get_mcause_data(
            phase="format_map",
            source=["COL_DANE", "ZAF_STATSSA", "ITA_ISTAT"],
            sub_dirs=subdirs,
            data_type_id=9, code_system_id=code_system_id,
            assert_all_available=True,
            verbose=True, **BLOCK_RERUN)
    else:
        udf = pd.DataFrame()

    print_log_message("reading in limited use data")
    datasets = get_datasets(**{"force_rerun": True, "block_rerun": False,
                               "source": MCauseLauncher.limited_sources,
                               "code_system_id": code_system_id})
    limited_metadata = datasets.apply(lambda x: str(
        x['nid']) + "_" + str(x['extract_type_id']), axis=1).values

    dfs = []
    for source in MCauseLauncher.limited_sources:
        limited_dir = get_limited_use_directory(source, int_cause, inj_garbage)
        csvfiles = glob.glob(os.path.join(limited_dir, "*.csv"))
        for file in csvfiles:
            if any(meta in file for meta in limited_metadata):
                df = pd.read_csv(file)
                dfs.append(df)
    ldf = pd.concat(dfs, ignore_index=True, sort=True)
    df = pd.concat([udf, ldf], sort=True, ignore_index=True)

    return df


def drop_age_restricted_cols(df):
    start = len(df)
    age_meta_df = get_ages(force_rerun=False, block_rerun=True)
    # secret causes in restrictions
    cause_meta_df = get_current_cause_hierarchy(cause_set_id=4,
            **{'block_rerun': True, 'force_rerun': False})
    injuries_restrictions = pd.read_csv("/homes/agesak/thesis/maps/injuries_overrides.csv")
    injuries_restrictions = add_cause_metadata(injuries_restrictions, add_cols='cause_id', merge_col='acause',
                            cause_meta_df=cause_meta_df)
    injuries_restrictions["age_start_group"] = injuries_restrictions["age_start_group"].fillna(0)


    orig_cols = df.columns
    df = add_age_metadata(
        df, add_cols=['age_group_years_start', 'age_group_years_end'],
        age_meta_df=age_meta_df
    )

    df = df.merge(injuries_restrictions, on='cause_id', how='left')

    # age_group_years_end is weird, 0-14 means age_group_years_end 15
    too_young = df["age_group_years_end"] <= df["age_start_group"]
    too_old = df["age_group_years_start"] > df["age_end_group"]

    df = df[~(too_young | too_old)]
    df = df[orig_cols]
    end = len(df)
    print_log_message(f"dropping {start - end} cols that violate age restrictions")

    return df

def create_train_test(df, test, int_cause, age_group_id):
    """Create train/test datasets, if running tests,
    randomly sample from all locations so models don't take forever to run"""
    locs = get_location_metadata(gbd_round_id=6, location_set_id=35)

    keep_cols = DEM_COLS + ["cause_info", f"{int_cause}"] + [
        x for x in list(df) if "multiple_cause" in x]

    df = df.loc[(df.age_group_id != 283) & (df.age_group_id != 160)]
    df = df[keep_cols]
    df = create_age_bins(df, AGG_AGES)
    df = drop_age_restricted_cols(df)
    if age_group_id:
        print_log_message(f"subsetting to just age group id {age_group_id}")
        df = df.loc[df["age_group_id"] == age_group_id]
        print_log_message(f"resulting df is {len(df)} rows")
    df["cause_age_info"] = df[["cause_info", "age_group_id"]].astype(
        str).apply(lambda x: " ".join(x), axis=1)
    df["dem_info"] = df[["cause_info", "location_id", "sex_id", "year_id", "age_group_id"]].astype(
        str).apply(lambda x: " ".join(x), axis=1)

    garbage_df = df.query(f"cause_id==743 & {int_cause}==1")
    df = df.query(f"cause_id!=743 & {int_cause}!=1")

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

    return train_df, test_df, garbage_df


def rf_params(model):
    assert model == "rf", "wrong model type"
    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    clf__estimator__n_estimators = df.loc[df[
        f"{model}"] == "clf__estimator__n_estimators",
        f"{model}_value"].str.split(",")[0]
    clf__estimator__max_depth = df.loc[df[
        f"{model}"] == "clf__estimator__max_depth",
        f"{model}_value"].str.split(",")[1]
    clf__estimator__max_features = df.loc[df[
        f"{model}"] == "clf__estimator__max_features",
        f"{model}_value"].str.split(",")[2]
    clf__estimator__criterion = df.loc[df[
        f"{model}"] == "clf__estimator__criterion",
        f"{model}_value"].str.split(",")[3]
    keys = "clf__estimator__n_estimators", "clf__estimator__max_depth", "clf__estimator__max_features", "clf__estimator__criterion"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        clf__estimator__n_estimators, clf__estimator__max_depth, clf__estimator__max_features, clf__estimator__criterion)]
    return params


def naive_bayes_params(model):
    assert (model == "multi_nb") | (model == "bernoulli_nb") | (
        model == "complement_nb"), "wrong model type"
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
    assert model == "svm", "wrong model type"
    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    clf__estimator__C = df.loc[df[
        f"{model}"] == "clf__estimator__C",
        f"{model}_value"].str.split(",")[0]
    clf__estimator__kernel = df.loc[df[
        f"{model}"] == "clf__estimator__kernel",
        f"{model}_value"].str.split(",")[1]
    clf__estimator__decision_function_shape = df.loc[df[
        f"{model}"] == "clf__estimator__decision_function_shape",
        f"{model}_value"].str.split(",")[2]
    clf__estimator__gamma = df.loc[df[
        f"{model}"] == "clf__estimator__gamma",
        f"{model}_value"].str.split(",")[3]
    clf__estimator__max_iter = df.loc[df[
        f"{model}"] == "clf__estimator__max_iter",
        f"{model}_value"].str.split(",")[4]
    keys = "clf__estimator__C", "clf__estimator__kernel", "clf__estimator__decision_function_shape", "clf__estimator__gamma", "clf__estimator__max_iter"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        clf__estimator__C, clf__estimator__kernel, clf__estimator__decision_function_shape, clf__estimator__gamma, clf__estimator__max_iter)]
    return params


def svm_bag_params(model):
    assert model == "svm_bag", "wrong model type"
    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    name__base_estimator__C = df.loc[df[
        f"{model}"] == "name__base_estimator__C",
        f"{model}_value"].str.split(",")[0]
    name__base_estimator__kernel = df.loc[df[
        f"{model}"] == "name__base_estimator__kernel",
        f"{model}_value"].str.split(",")[1]
    name__base_estimator__decision_function_shape = df.loc[df[
        f"{model}"] == "name__base_estimator__decision_function_shape",
        f"{model}_value"].str.split(",")[2]
    name__base_estimator__max_iter = df.loc[df[
        f"{model}"] == "name__base_estimator__max_iter",
        f"{model}_value"].str.split(",")[3]
    name__n_estimators = df.loc[df[
        f"{model}"] == "name__n_estimators",
        f"{model}_value"].str.split(",")[4]
    name__max_samples = df.loc[df[
        f"{model}"] == "name__max_samples",
        f"{model}_value"].str.split(",")[5]
    name__bootstrap_features = df.loc[df[
        f"{model}"] == "name__bootstrap_features",
        f"{model}_value"].str.split(",")[6]
    name__oob_score = df.loc[df[
        f"{model}"] == "name__oob_score",
        f"{model}_value"].str.split(",")[7]
    keys = "name__base_estimator__C", "name__base_estimator__kernel", "name__base_estimator__decision_function_shape", "name__base_estimator__max_iter", "name__n_estimators", "name__max_samples", "name__bootstrap_features", "name__oob_score"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        name__base_estimator__C, name__base_estimator__kernel, name__base_estimator__decision_function_shape, name__base_estimator__max_iter, name__n_estimators, name__max_samples, name__bootstrap_features, name__oob_score)]
    return params


def gbt_params(model):
    assert model == "gbt", "wrong model type"
    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    clf__estimator__n_estimators = df.loc[df[
        f"{model}"] == "clf__estimator__n_estimators",
        f"{model}_value"].str.split(",")[0]
    clf__estimator__learning_rate = df.loc[df[
        f"{model}"] == "clf__estimator__learning_rate",
        f"{model}_value"].str.split(",")[1]
    clf__estimator__max_depth = df.loc[df[
        f"{model}"] == "clf__estimator__max_depth",
        f"{model}_value"].str.split(",")[2]
    clf__estimator__max_features = df.loc[df[
        f"{model}"] == "clf__estimator__max_features",
        f"{model}_value"].str.split(",")[3]
    keys = "clf__estimator__n_estimators", "clf__estimator__learning_rate", "clf__estimator__max_depth", "clf__estimator__max_features"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        clf__estimator__n_estimators, clf__estimator__learning_rate, clf__estimator__max_depth, clf__estimator__max_features)]
    return params


def xgb_params(model):
    assert model == "xgb", "wrong model type"
    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    clf__estimator__eta = df.loc[df[
        f"{model}"] == "clf__estimator__eta",
        f"{model}_value"].str.split(",")[0]
    clf__estimator__gamma = df.loc[df[
        f"{model}"] == "clf__estimator__gamma",
        f"{model}_value"].str.split(",")[1]
    clf__estimator__max_depth = df.loc[df[
        f"{model}"] == "clf__estimator__max_depth",
        f"{model}_value"].str.split(",")[2]
    clf__estimator__subsample = df.loc[df[
        f"{model}"] == "clf__estimator__subsample",
        f"{model}_value"].str.split(",")[3]
    clf__estimator__n_estimators = df.loc[df[
        f"{model}"] == "clf__estimator__n_estimators",
        f"{model}_value"].str.split(",")[4]
    keys = "clf__estimator__eta", "clf__estimator__gamma", "clf__estimator__max_depth", "clf__estimator__subsample", "clf__estimator__n_estimators"
    params = [dict(zip(keys, combo)) for combo in itertools.product(
        clf__estimator__eta, clf__estimator__gamma, clf__estimator__max_depth, clf__estimator__subsample, clf__estimator__n_estimators)]
    return params


def format_argparse_params(param, param_len):
    assert len(param) == param_len, "error.. more than one set of params"
    # turn all params into a single "_" separated string for argparse
    param = "_".join([str(x) for x in list(param.values())])
    return param
