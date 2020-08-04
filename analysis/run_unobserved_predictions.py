import pandas as pd
import sys
import six

from thesis_utils.misc import str2bool
from thesis_utils.modeling import create_neural_network
from cod_prep.utils.misc import print_log_message

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBClassifier

## add for quick run
from thesis_utils.model_evaluation import get_best_fit
from cod_prep.claude.claude_io import makedirs_safely

DEM_COLS = ["age_group_id", "sex_id", "location_id", "year_id"]


def read_in_summary_stats(model_path):
    """Read in and append all summary stats files across 500 test datasets"""

    summaries = []
    for dataset_num in range(1, 501, 1):
        df = pd.read_csv(
            f"{model_path}/dataset_{dataset_num}_summary_stats.csv")
        summaries.append(df)
    return pd.concat(summaries)


def aggregate_evaluation_metrics(summaries, testing_dir):
    """Generate mean, median, max, and min for
    evaluation metrics across 500 test datasets"""
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
    summary_df = pd.concat(
        [mean, median, maximum, minimum], axis=1).reset_index(
    ).rename(columns={"index": "Evaluation metrics"})
    summary_df.to_csv(f"{testing_dir}/model_metrics_summary.csv", index=False)


def main(data_dir, predicted_test_dir, int_cause, short_name,
         model_name, age_feature, dem_feature):
    """Summarize evaluation metrics across 500 test datasets
       Refit the classifier on all observed data
       Predict on the unobserved data
    """

    # determine the model's feature vector
    if age_feature:
        x_col = "cause_age_info"
    elif dem_feature:
        x_col = "dem_info"
    else:
        x_col = "cause_info"

    ## comment out for quick run
    ## summaries = read_in_summary_stats(predicted_test_dir)

    ## comment out for quick run
    ## summarize evaluation metrics across the datasets
    ## aggregate_evaluation_metrics(summaries, predicted_test_dir)

    # read in test df
    test_df = pd.read_csv(
        f"{data_dir}/test_df.csv")[DEM_COLS + ["cause_id",
                                               f"{x_col}",
                                               f"{int_cause}"]]
    # read in train df
    train_df = pd.read_csv(
        f"{data_dir}/train_df.csv")[DEM_COLS + ["cause_id",
                                                f"{x_col}",
                                                f"{int_cause}"]]
    print_log_message("read in train and test")
    # concat train/test to refit a model on all the observed data
    df = pd.concat([train_df, test_df], sort=True, ignore_index=True)

    print_log_message("reading in params df")
    param_df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    param_df = param_df[[x for x in list(param_df) if short_name in x]]
    param_df[f"{short_name}"] = param_df[f"{short_name}"].str.replace(
        "clf__estimator__", "")
    ## comment out for quick run
    ## params = summaries.best_model_params.iloc[0]
    ## add for quick run
    params = get_best_fit(data_dir, short_name)

    # format best params to feed to classifier
    if isinstance(params, six.string_types):
        best_params = params.split("_")
    else:
        best_params = [params]

    param_kwargs = dict(zip(param_df.iloc[:, 0], best_params))
    if short_name == "nn":
        # these feed into create_neural_network
        hidden_nodes_1 = int(param_kwargs["hidden_nodes_1"])
        hidden_layers = int(param_kwargs["hidden_layers"])
        hidden_nodes_2 = int(param_kwargs["hidden_nodes_2"])
        # parameters with clf__ are only fed to keras classifier
        param_kwargs = {k: v for k, v in param_kwargs.items() if "clf__" in k}

    # ensure column dtypes are correct
    measure_dict = {"int": int, "float": float, "str": str}
    for key, value in param_kwargs.items():
        dtype = param_df.loc[param_df[
            f"{short_name}"] == key, f"{short_name}_dtype"].iloc[0]
        param_kwargs[key] = measure_dict[dtype](param_kwargs[key])

    # run Neural network separately because classifier
    # takes secondary arguments related to build
    if short_name == "nn":
        param_kwargs = {k.replace("clf__", ""): v for k,
                        v in param_kwargs.items() if "clf__" in k}
        cv = CountVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b")
        tf = cv.fit_transform(df[f"{x_col}"])
        print_log_message("converting to dense matrix")
        tf = tf.todense()
        # just hard code classifer name because this only works for keras
        model = KerasClassifier(build_fn=create_neural_network,
                                output_nodes=len(
                                    df.cause_id.unique()),
                                hidden_layers=hidden_layers,
                                hidden_nodes_1=hidden_nodes_1,
                                hidden_nodes_2=hidden_nodes_2, **param_kwargs)
        print_log_message("fitting KerasClassifier")
        model.fit(tf, df["cause_id"].values, **param_kwargs)
    else:
        # refit all other classifiers
        cv = CountVectorizer(lowercase=False)
        tf = cv.fit_transform(df[f"{x_col}"])
        print_log_message(f"fitting {model_name}")
        model = eval(model_name)(**param_kwargs).fit(tf, df["cause_id"])

    # now predict on the unobserved data
    print_log_message("reading in unobserved_df")

    unobserved_df = pd.read_csv(
        f"{data_dir}/int_cause_df.csv")[DEM_COLS + ["cause_id",
                                                    f"{x_col}",
                                                    f"{int_cause}"]]
    new_counts = cv.transform(unobserved_df[f"{x_col}"])
    if short_name == "nn":
        print_log_message("converting unobserved data to dense matrix")
        new_counts = new_counts.todense()
    unobserved_df["predictions"] = model.predict(new_counts)

    ## add for quick run
    makedirs_safely(predicted_test_dir)

    print_log_message("writing to df")
    unobserved_df.to_csv(f"{predicted_test_dir}/model_predictions.csv")
    joblib.dump(
        model, f"{predicted_test_dir}/model_fit.pkl")
    print_log_message("wrote model fit")


if __name__ == '__main__':

    data_dir = str(sys.argv[1])
    predicted_test_dir = str(sys.argv[2])
    int_cause = str(sys.argv[3])
    short_name = str(sys.argv[4])
    model_name = str(sys.argv[5])
    age_feature = str2bool(sys.argv[6])
    dem_feature = str2bool(sys.argv[7])

    main(data_dir, predicted_test_dir, int_cause,
         short_name, model_name, age_feature, dem_feature)
