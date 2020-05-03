import pandas as pd
import sys
import os
import six

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB

DEM_COLS = ["age_group_id", "sex_id", "location_id", "year_id"]


def read_in_summary_stats(model_path):

    summaries = []
    for dataset_num in range(1, 501, 1):
        df = pd.read_csv(
            f"{model_path}/dataset_{dataset_num}_summary_stats.csv")
        summaries.append(df)
    return pd.concat(summaries)


def aggregate_evaluation_metrics(summaries, testing_dir):

    # FIX NEW METRICS
    """Generate mean, median, max, and min for evaluation metrics across 500 test datasets"""
    metric_cols = [x for x in list(summaries) if x != "best_model_params"]
    # summarize evaluation metrics across the datasets
    mean = summaries[metric_cols].apply(
        lambda x: x.mean(axis=0)).rename("Mean")
    median = summaries[metric_cols].apply(
        lambda x: x.median(axis=0)).rename("Median")
    maximum = summaries[metric_cols].apply(
        lambda x: x.max(axis=0)).rename("Max")
    minimum = summaries[metric_cols].apply(
        lambda x: x.min(axis=0)).rename("Min")
    # write to df
    summary_df = pd.concat([mean, median, maximum, minimum], axis=1).reset_index(
    ).rename(columns={"index": "Evaluation metrics"})
    summary_df.to_csv(f"{testing_dir}/model_metrics_summary.csv", index=False)


def main(data_dir, predicted_test_dir, int_cause, short_name, model_name):
    # read in predictions from 500 test datasets
    # aggregate the evaluation metrics from each of the 500
    # refit model on all the observed data
    # predict on the x59/y34 data

    summaries = read_in_summary_stats(predicted_test_dir)

    # summarize evaluation metrics across the datasets
    aggregate_evaluation_metrics(summaries, predicted_test_dir)

    # read in test and train df
    test_df = pd.read_csv(
        f"{data_dir}/test_df.csv")[DEM_COLS + ["cause_id", "cause_info", f"{int_cause}"]]
    train_df = pd.read_csv(
        f"{data_dir}/test_df.csv")[DEM_COLS + ["cause_id", "cause_info", f"{int_cause}"]]
    print("read in train and test")
    df = pd.concat([train_df, test_df], sort=True, ignore_index=True)

    # refit a model on all the observed data
    print("reading in params df")
    param_df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    param_df = param_df[[x for x in list(param_df) if short_name in x]]
    # param_df[f"{short_name}"] = param_df[f"{short_name}"].str.replace(
    #     "clf__estimator__", "")

    # IDK IF THIS IF STATEMENTWORKS
    if short_name == "svm_bag":
        param_df = param_df.loc[param_df[f"{short_name}"].str.contains("base_estimator")]
        param_df[f"{short_name}"] = param_df[f"{short_name}"].str.replace(
            "name__base_estimator__", "")
    else:
        param_df[f"{short_name}"] = param_df[f"{short_name}"].str.replace(
            "clf__estimator__", "")        
    params = summaries.best_model_params.iloc[0]
    if isinstance(params, six.string_types):
        best_params = params.split("_")
    else:
        best_params = [params]

    # param_names = param_df.iloc[:,0]
    param_kwargs = dict(zip(param_df.iloc[:, 0], best_params))
    # ensure column dtypes are correct
    measure_dict = {"int": int, "float": float, "str":str}
    for key, value in param_kwargs.items():
        dtype = param_df.loc[param_df[f"{short_name}"] == key, f"{short_name}_dtype"].iloc[0]
        param_kwargs[key] = measure_dict[dtype](param_kwargs[key])

    # do the refit
    cv = CountVectorizer()
    tf = cv.fit_transform(df["cause_info"])
    model_fit = eval(model_name)(**param_kwargs).fit(tf, df["cause_id"])

    # now predict on the unobserved data
    print("reading in unobserved_df")
    unobserved_df = pd.read_csv(
        f"{data_dir}/int_cause_df.csv")[DEM_COLS + ["cause_id", "cause_info", f"{int_cause}"]]
    # need to remember explicitly what this does
    new_counts = cv.transform(unobserved_df["cause_info"])
    unobserved_df["predictions"] = model_fit.predict(new_counts)

    # do I wanna save anything else about this?
    # also this might not be the best place to save this
    print("writing to df")
    unobserved_df.to_csv(f"{predicted_test_dir}/model_predictions.csv")
    joblib.dump(
        model_fit, f"{predicted_test_dir}/model_fit.csv")
    print("wrote model fit")


if __name__ == '__main__':

    data_dir = str(sys.argv[1])
    predicted_test_dir = str(sys.argv[2])
    int_cause = str(sys.argv[3])
    short_name = str(sys.argv[4])
    model_name = str(sys.argv[5])
    print(data_dir)
    print(predicted_test_dir)

    main(data_dir, predicted_test_dir, int_cause, short_name, model_name)
