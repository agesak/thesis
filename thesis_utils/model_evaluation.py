import pandas as pd
import numpy as np
import os

from cod_prep.claude.configurator import Configurator
from mcod_prep.utils.causes import get_most_detailed_inj_causes

CONF = Configurator('standard')

def calculate_cccsmfa(y_true, y_pred):
    """ Calculate chance-corrected cause-specific mortality fraction accuracy
    built from https://github.com/aflaxman/siaman16-va-minitutorial/
                blob/master/1-tutorial-notebooks/4-va_csmf.ipynb
    Arguments:
        y_true: array of true cause ids
        y_pred: array of predicted cause ids
    """
    random_allocation = 0.632

    csmf_true = pd.Series(y_true).value_counts() / float(len(y_true))
    csmf_pred = pd.Series(y_pred).value_counts() / float(len(y_pred))
    numerator = np.abs(csmf_true - csmf_pred)
    # first get csmfa
    csmfa = 1 - (numerator.sum()) / (2 * (1 - np.min(csmf_true)))

    # then get cccsmfa
    cccsmfa = (csmfa - random_allocation) / (1 - random_allocation)

    return cccsmfa


def calculate_concordance(y_true, y_pred, int_cause):
    """Calculate chance-corrected concordance
    Equation: ((TP/TP+FN) - 1/N)/(1 - 1/N)
    Arguments:
        y_true: array of true cause ids
        y_pred: array of predicted cause ids
        int_cause: intermediate cause of interest (x59 or y34)
    """

    # get an array of the most detailed injuries cause_ids in GBD
    causes = np.array(get_most_detailed_inj_causes(int_cause,
        cause_set_version_id=CONF.get_id('reporting_cause_set_version'),
        **{'block_rerun': True, 'force_rerun': False}))

    for cause in causes:
        # TP+FN - the number of deaths for a cause
        denom = (y_true == cause).sum(axis=0)
        # TP/denom
        total = ((y_true == cause) & (y_pred == cause)).sum(axis=0) / denom
        # chance-corrected concordance
        ccc = (total - (1 / len(causes))) / (1 - (1 / len(causes)))
        causes = np.where(causes == cause, ccc, causes)
    # it's possible some causes have zero rows in a given test dataset
    cccc = np.nanmean(causes, axis=0)

    return cccc


def get_best_fit(model_dir, short_name):
    """Use the CCC to decide which model performs the best
    Arguments:
        model_dir: (str) parent directory where all models are located
        short_name: (str) abbreviated ML classifier names -
                    defined in thesis_analysis.launch_models
    """
    dfs = []
    for root, dirs, files in os.walk(os.path.join(
            os.path.join(model_dir, short_name))):
        for stats_dir in dirs:
            df = pd.read_csv(os.path.join(
                model_dir, short_name, stats_dir, "summary_stats.csv"))
            df["model_params"] = stats_dir
            dfs.append(df)
    df = pd.concat(dfs, sort=True, ignore_index=True)
    df = df.sort_values(by="mean_test_concordance",
                                  ascending=False)[["model_params"]]
    best_fit = df.iloc[0,0].replace("country_model_", "")

    return best_fit


# def format_best_fit_params(best_fit, model_name):
#     """Format a set of model parameters to run in the GridSearchCV pipeline
#     Arguments:
#         best_fit: df with cv_results_ method of GridSearchCV output
#     """
#     df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")

#     best_fit = best_fit[[x for x in list(best_fit) if (
#         "param_" in x) & ~(x.endswith("estimator"))]].dropna(axis=1)
#     best_fit.columns = best_fit.columns.str.lstrip('param_')
#     # format the parameterts
#     params = []
#     for col in list(best_fit):
#         params = params + best_fit[col].values.tolist()
#     param_dict = dict(zip(list(best_fit), params))
#     # merge on parameter df to ensure correct order
#     df = df.merge(pd.DataFrame.from_dict(
#         param_dict, orient="index").reset_index(
#     ).rename(columns={"index": model_name}), on=model_name)
#     # could be a problem in other places... basically float rows convert whole row to float.. which isnt what i want
#     if model_name == "GradientBoostingClassifier":
#         df[0] = np.where(df["GradientBoostingClassifier_dtype"] == "int", df[0].apply(int).astype(str), df[0])
#     best_model_params = "_".join(
#         df[0].dropna().astype(str).values.tolist())
#     # best_model_params = format_argparse_params(model_name, params)

#     return best_model_params

def format_for_bow(df, age_feature, dem_feature):
    keep_cols = ["cause_id", "cause_info"]
    multiple_cause_cols = [x for x in list(df) if "cause" in x]
    multiple_cause_cols.remove("cause_id")
    df["cause_info"] = df[[x for x in list(
        df) if "multiple_cause" in x]].fillna(
        "").astype(str).apply(lambda x: " ".join(x), axis=1)
    if age_feature:
        df["cause_age_info"] = df[["cause_info", "age_group_id"]].astype(
        str).apply(lambda x: " ".join(x), axis=1)
        keep_cols += ["cause_age_info"]
    if dem_feature:
        df["dem_info"] = df[["cause_info", "location_id", "sex_id", "year_id", "age_group_id"]].astype(
        str).apply(lambda x: " ".join(x), axis=1)
        keep_cols += ["dem_info"]
    df = df[keep_cols]
    return df


def generate_multiple_cause_rows(sample_df, test_df, cause, age_feature, dem_feature):
    """
    Arguments:
        sample_df: cause-specific df with number of rows equal to
                   cause-specific proportion from dirichlet
        test_df: true test df, corresponding to 25% of data
        cause: injuries cause of interest
    Returns:
        cause-specific df with chain cols randomly sampled from test df
    """

    # get single "cause_info" column
    if age_feature:
        x_col = "cause_age_info"
    elif dem_feature:
        x_col = "dem_info"
    else:
        x_col = "cause_info"
    test_df = format_for_bow(test_df, age_feature, dem_feature)

    # subset to only cause-specific rows in test df
    cause_df = test_df.loc[test_df.cause_id == cause]
    # drop any na rows
    assert len(cause_df) != 0, "subsetting test df failed in creating 500 datasets"
    # assign chain causes by randomly sampling (with replacement) rows of cause-specific test df
    print("about to sample test df")
    sample_df = cause_df[[f"{x_col}"]].sample(
        len(sample_df), replace=True).reset_index(drop=True)
    print("finished sampling")
    sample_df["cause_id"] = cause

    return sample_df
