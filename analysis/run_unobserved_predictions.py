import pandas as pd
import sys
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB

DEM_COLS = ["age_group_id", "sex_id", "location_id", "year_id"]


def read_in_summary_stats(model_path):
    summaries = []
    for r, d, f in os.walk(model_path):
        for file in f:
            if '.csv' in file:
                summary_stats = pd.read_csv(os.path.join(r, file))
                summaries.append(summary_stats)
    return pd.concat(summaries)


def aggregate_evaluation_metrics(summaries, testing_dir):
    """Generate mean, median, max, and min for evaluation metrics across 500 test datasets"""
    metric_cols = ["concordance", "cccsmfa", "recall", "precision"]
    # summarize evaluation metrics across the datasets
    mean = summaries[metric_cols].apply(
        lambda x: x.mean(axis=0)).rename("mean")
    median = summaries[metric_cols].apply(
        lambda x: x.median(axis=0)).rename("median")
    maximum = summaries[metric_cols].apply(
        lambda x: x.max(axis=0)).rename("max")
    minimum = summaries[metric_cols].apply(
        lambda x: x.min(axis=0)).rename("min")
    # write to df
    summary_df = pd.concat([mean, median, maximum, minimum], axis=1).reset_index(
    ).rename(columns={"index": "Evaluation metrics"})
    summary_df.to_csv(f"{testing_dir}/model_metrics_summary.csv", index=False)


def main(testing_dir, data_dir, int_cause, model_name):
    # read in predictions from 500 test datasets
    # aggregate the evaluation metrics from each of the 500
    # refit model on all the observed data
    # predict on the x59/y34 data

    # ex.
    # testing_dir = "/ihme/cod/prep/mcod/process_data/x59/thesis/sample_dirichlet/gbt/2020_04_14_test/300_0.1"
    # data_dir = "/ihme/cod/prep/mcod/process_data/x59/thesis/2020_04_14_test"
    summaries = read_in_summary_stats(testing_dir)

    # summarize evaluation metrics across the datasets
    aggregate_evaluation_metrics(summaries)

    # read in test and train df
    test_df = pd.read_csv(
        f"{data_dir}/test_df.csv")[DEM_COLS + ["cause_id", "cause_info", f"{int_cause}"]]
    train_df = pd.read_csv(
        f"{data_dir}/test_df.csv")[DEM_COLS + ["cause_id", "cause_info", f"{int_cause}"]]
    df = pd.concat([train_df, test_df], sort=True, ignore_index=True)

    # refit a model on all the observed data
    param_df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    param_df = param_df[[x for x in list(param_df) if model_name in x]]
    param_df[f"{model_name}"] = param_df[f"{model_name}"].str.replace(
        "clf__estimator__", "")
    best_params = summaries.best_model_params.iloc[0].split("_")

    # param_names = param_df.iloc[:,0]
    param_kwargs = dict(zip(param_df.iloc[:, 0], best_params))
    # ensure column dtypes are correct
    measure_dict = {"int": int, "float": float}
    for key, value in param_kwargs.items():
        dtype = param_df.loc[param_df[f"{model_name}"] == key, f"{model_name}_dtype"].iloc[0]
        param_kwargs[key] = measure_dict[dtype](param_kwargs[key])

    # do the refit
    cv = CountVectorizer()
    tf = cv.fit_transform(df["cause_info"])
    model_fit = eval(model_name)(**param_kwargs).fit(tf, df["cause_id"])

    # now predict on the unobserved data
    unobserved_df = pd.read_csv(
        f"{data_dir}/int_cause_df.csv")[DEM_COLS + ["cause_id", "cause_info", f"{int_cause}"]]
    # need to remember explicitly what this does
    new_counts = cv.transform(unobserved_df["cause_info"])
    unobserved_df["predictions"] = model_fit.predict(new_counts)

    # do I wanna save anything else about this?
    # also this might not be the best place to save this
    unobserved_df.to_csv(f"{testing_dir}/model_predictions.csv")


if __name__ == '__main__':

    testing_dir = str(sys.argv[1])
    data_dir = str(sys.argv[2])
    int_cause = int(sys.argv[3])
    model_name = int(sys.argv[4])

    main(testing_dir, data_dir, int_cause, model_name)
