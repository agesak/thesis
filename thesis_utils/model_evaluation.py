import pandas as pd
import numpy as np
import os

from cod_prep.claude.claude_io import makedirs_safely
from mcod_prep.utils.causes import get_most_detailed_inj_causes


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
def get_best_fit(model_dir, short_name):
    """Use the CCCSMFA to decide which model performs the best
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
            dfs.append(df)
    df = pd.concat(dfs, sort=True, ignore_index=True)

    # IDK WHAT ASCENDING SHOULD BE HERE BECAUSE IT'S NEGATIVE
    best_fit = df.sort_values(by="mean_test_cccsfma",
                              ascending=False).reset_index(drop=True).iloc[0:1]
    return best_fit


def format_best_fit_params(best_fit, model_name):
    """Format a set of model parameters to run in the GridSearchCV pipeline
    Arguments:
        best_fit: df with cv_results_ method of GridSearchCV output
    """
    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")

    best_fit = best_fit[[x for x in list(best_fit) if (
        "param_" in x) & ~(x.endswith("estimator"))]].dropna(axis=1)
    best_fit.columns = best_fit.columns.str.lstrip('param_')
    # format the parameterts
    params = []
    for col in list(best_fit):
        params = params + best_fit[col].values.tolist()
    param_dict = dict(zip(list(best_fit), params))
    # merge on parameter df to ensure correct order
    df = df.merge(pd.DataFrame.from_dict(
        param_dict, orient="index").reset_index(
    ).rename(columns={"index": model_name}), on=model_name)
    best_model_params = "_".join(
        df[0].dropna().astype(int).astype(str).values.tolist())
    # best_model_params = format_argparse_params(model_name, params)

    return best_model_params

# IS THIS EVEN RIGHT? lol


def generate_multiple_cause_rows(sample_df, test_df, cause):
    """
    Arguments:
        sample_df: cause-specific df with number of rows equal to
                   cause-specific proportion from dirichlet
        test_df: true test df, corresponding to 25% of data
        cause: injuries cause of interest
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

    # drop these columns so .iloc will work
    sample_df.drop(columns=["cause", "cause_id"], inplace=True)
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

# IS THIS EVEN RIGHT? lol


def create_testing_datasets(test_df, write_dir, num_datasets=500,
                            df_size=1000):

    # dictionary of causes and their respective proportions in the data
    cause_distribution = test_df['cause_id'].value_counts(
        normalize=True).to_dict()
    # 500 dirichlet distributions based on test data cause distribution
    dts = np.random.dirichlet(alpha=list(
        cause_distribution.values()), size=num_datasets)

    # such a random guess..
    # should these be the length of the actual test df?
    df_size = df_size

    datasets = [np.NaN] * len(dts)
    for i in range(0, len(dts)):
        df_dir = f"{write_dir}/dataset_{i+1}"
        makedirs_safely(df_dir)
        tdf = pd.DataFrame({"cause": [np.NaN] * df_size})
        # dictionary of cause ids to each dirichlet distribution
        cd = dict(zip(cause_distribution.keys(), dts[i]))
        df = []
        for cause in cd.keys():
            print(f"{cause}_{i+1}")
            # proportion from dirichlet dictates how many rows are assigned to a given cause
            s_tdf = tdf.sample(
                frac=cd[cause], replace=False).assign(cause_id=cause)
            s_tdf = generate_multiple_cause_rows(s_tdf, test_df, cause)
            df.append(s_tdf)
        all_h = pd.concat(df, ignore_index=True, sort=True)
        # # compare
        # df['cause'].value_counts(normalize=True).to_dict()
        # # to - these fractions aren't alwayss the same
        # cd
        datasets[i] = all_h
        all_h.to_csv(f"{df_dir}/dataset.csv", index=False)
