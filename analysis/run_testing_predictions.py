"""predict on test datasets"""
import pandas as pd
import sys
from thesis_utils.model_evaluation import (calculate_cccsmfa,
                                           calculate_concordance)

from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, accuracy_score


def format_for_bow(df):
    multiple_cause_cols = [x for x in list(df) if "cause" in x]
    multiple_cause_cols.remove("cause_id")
    # maybe try "" here
    df["cause_info"] = df[[x for x in list(
        df) if "multiple_cause" in x]].fillna(
        "").astype(str).apply(lambda x: " ".join(x), axis=1)
    df = df[["cause_id", "cause_info"]]
    return df


def main(best_model_dir, dataset_dir, testing_model_dir, best_model_params, int_cause, dataset_num):

    # read in model object of best model
    grid_results = joblib.load(f"{best_model_dir}/grid_results.pkl")

    # read in test dataset
    dataset = pd.read_csv(f"{dataset_dir}/dataset_{dataset_num}.csv")
    dataset = format_for_bow(dataset)

    # predit on test dataset
    dataset["predicted"] = grid_results.predict(dataset["cause_info"])


    macro_precision = precision_score(y_true=dataset.cause_id,
                                      y_pred=dataset.predicted, average="macro")
    micro_precision = precision_score(y_true=dataset.cause_id,
                                      y_pred=dataset.predicted, average="micro")
    macro_recall = recall_score(y_true=dataset.cause_id,
                                y_pred=dataset.predicted, average="macro")
    micro_recall = recall_score(y_true=dataset.cause_id,
                                y_pred=dataset.predicted, average="micro")
    accuracy = accuracy_score(y_true=dataset.cause_id,
                              y_pred=dataset.predicted)
    cccsmfa = calculate_cccsmfa(y_true=dataset.cause_id,
                                y_pred=dataset.predicted)
    concordance = calculate_concordance(y_true=dataset.cause_id,
                                        y_pred=dataset.predicted,
                                        int_cause=int_cause)

    # maybe save something identifiable about model
    df = pd.DataFrame({"Concordance": [concordance],
                       "CCCSMFA": [cccsmfa],
                       "Macro Recall": [macro_recall],
                       "Micro Recall": [micro_recall],
                       "Macro Precision": [macro_precision],
                       "Micro Precision": [micro_precision],
                       "Accuracy": [accuracy],
                       "best_model_params": [best_model_params]})
    df.to_csv(
        f"{testing_model_dir}/dataset_{dataset_num}_summary_stats.csv", index=False)
    dataset.to_csv(
        f"{testing_model_dir}/dataset_{dataset_num}_predictions.csv", index=False)
    joblib.dump(
        grid_results, f"{testing_model_dir}/dataset_{dataset_num}_grid_results.pkl")


if __name__ == '__main__':

    best_model_dir = str(sys.argv[1])
    dataset_dir = str(sys.argv[2])
    testing_model_dir = str(sys.argv[3])
    best_model_params = str(sys.argv[4])
    int_cause = str(sys.argv[5])
    dataset_num = int(sys.argv[6])

    print(best_model_dir)
    print(dataset_dir)
    print(testing_model_dir)
    print(best_model_params)
    print(int_cause)
    print(dataset_num)

    main(best_model_dir, dataset_dir, testing_model_dir,
         best_model_params, int_cause, dataset_num)
