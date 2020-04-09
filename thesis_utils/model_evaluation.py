import pandas as pd
import numpy as np
import os
import argparse

from cod_prep.claude.claude_io import makedirs_safely
from mcod_prep.utils.causes import get_most_detailed_inj_causes


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


def calculate_cccsmfa(y_true, y_pred):
    # https://stackoverflow.com/questions/32401493/how-to-create-customize-your-own-scorer-function-in-scikit-learn
    # built from https://github.com/aflaxman/siaman16-va-minitutorial/blob/master/1-tutorial-notebooks/4-va_csmf.ipynb
    # y true is true cause ids
    # y pred is predicted cause ids

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
    ((TP/TP+FN) - 1/N)/(1 - 1/N)"""

    # get an array of the cause_ids in my data
    causes = np.array(get_most_detailed_inj_causes(int_cause, cause_set_id=4))

    for cause in causes:
        # TP+FN - the number of deaths for a cause
        denom = (y_true == cause).sum(axis=0)
        # TP/denom
        total = ((y_true == cause) & (y_pred == cause)).sum(axis=0) / denom
        # chance-corrected concordance
        ccc = (total - (1 / len(causes))) / (1 - (1 / len(causes)))
        causes = np.where(causes == cause, ccc, causes)

    cccc = np.mean(causes, axis=0)

    return cccc


# FIGURING THIS OUT STILL
# read in and append all summary files for a given model type
# pick model parameters with highest value for a given precision metric
def get_best_fit(model_dir, short_model):
    dfs = []
    for root, dirs, files in os.walk(os.path.join(os.path.join(model_dir, short_model))):
        for stats_dir in dirs:
            df = pd.read_csv(os.path.join(
                model_dir, short_model, stats_dir, "summary_stats.csv"))
            dfs.append(df)
    df = pd.concat(dfs, sort=True, ignore_index=True)

    # idk what ascending should be here bc it's negative?
    best_fit = df.sort_values(by="mean_test_cccsfma",
                              ascending=False).reset_index(drop=True).iloc[0:1]

    # should i have been saving the model object?
    # saves as pipeline object
    return best_fit

def format_best_fit_params(best_fit):

    best_fit = best_fit[[x for x in list(best_fit) if ("param_" in x) & ~(x.endswith("estimator"))]].dropna(axis=1)
    # format the parameterts
    params = []
    for col in list(best_fit):
        params = params + best_fit[col].values.tolist()
    params = [str(x) for x in params]
    params = "_".join(params)  
    best_model_params = format_params(model_name, params)   

    return best_model_params

def generate_multiple_cause_rows(sample_df, test_df, cause):
    """
    Arguments:
        sample_df: cause-specific df with number of rows equal to cause-specific proportion from dirichlet
        test_df: true test df, corresponding to 25% of data
        cause: cause of interest
    Returns:
        cause-specific df with chain cols randomly sampled from test df
    """
    multiple_cause_cols = [x for x in list(test_df) if "multiple_cause" in x]

    # create multiple cause columns in the sample df
    sample_df = pd.concat([sample_df, pd.DataFrame(
        columns=multiple_cause_cols)], sort=True, ignore_index=True)
    # subset to only cause-specific rows in test df
    cause_df = test_df.loc[test_df.cause_id == cause]
    assert len(cause_df) != 0, "subsetting test df failed in creating 500 datasets"

    # drop this column so .iloc will work
    sample_df.drop(columns="cause_id", inplace=True)
    # loop through rows of sample df
    for index, row in sample_df.iterrows():
        # should I be worried about replacement here?
        # randomly sample 1 row in the cause-specific test df
        chain = cause_df[multiple_cause_cols].sample(1).iloc[0]
        # assign the multiple cause cols in the sample df to these chain cols
        sample_df.iloc[[index], :] = chain.values
    # add this column back
    sample_df["cause_id"] = cause
    return sample_df


def create_testing_datsets(test_df, write_dir):

    # dictionary of causes and their respective proportions in the data
    cause_distribution = test_df['cause_id'].value_counts(
        normalize=True).to_dict()
    # 500 dirichlet distributions based on test data cause distribution
    dts = np.random.dirichlet(alpha=list(
        cause_distribution.values()), size=500)

    # such a random guess..
    # should these be the length of the actual test df?
    df_size = 1000

    datasets = [np.NaN] * len(dts)
    for i in range(0, len(dts)):
        write_dir = f"{write_dir}/dataset_{i+1}"
        makedirs_safely(write_dir)
        tdf = pd.DataFrame({"cause": [np.NaN] * df_size})
        # dictionary of cause ids (order preserved i think?) to each dirichlet distribution
        cd = dict(zip(cause_distribution.keys(), dts[i]))
        df = []
        for cause in cd.keys():
            # proportion from dirichlet dictates how many rows are assigned to a given cause
            s_tdf = tdf.sample(
                frac=cd[cause], replace=False).assign(cause_id=cause)
            s_tdf = generate_multiple_cause_rows(s_tdf, test_df, cause)
            df.append(s_tdf)
        all_h = pd.concat(df, ignore_index=True, sort=True)
        # compare
        df['cause'].value_counts(normalize=True).to_dict()
        # to - these fractions aren't alwayss the same
        cd
        datasets[i] = all_h

        all_h.to_csv(f"{write_dir}/dataset.csv", index=False)
