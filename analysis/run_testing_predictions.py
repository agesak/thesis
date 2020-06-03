"""predict on test datasets"""
import pandas as pd
import sys
import dill

from cod_prep.utils.misc import print_log_message
from thesis_utils.model_evaluation import (calculate_cccsmfa,
                                           calculate_concordance)
from thesis_utils.misc import str2bool

from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, accuracy_score

import theano
# get INFO (theano.gof.compilelock): Waiting for existing lock by unknown process (I am process '16768')
# errors for launching in parallel without this
theano.gof.compilelock.set_lock_status(False)


def main(best_model_dir, dataset_dir, testing_model_dir, best_model_params, int_cause, dataset_num, age_feature, dem_feature):
  """Predict on each test dataset"""
  if age_feature:
    x_col = "cause_age_info"
  elif dem_feature:
    x_col = "dem_info"
  else:
    x_col = "cause_info"
  # read in model object of best models
  print_log_message("reading in grid results object")
  grid_results = joblib.load(f"{best_model_dir}/grid_results.pkl")

  # read in test dataset
  print_log_message("reading in data")
  dataset = pd.read_csv(f"{dataset_dir}/dataset_{dataset_num}.csv")

  # predit on test dataset
  print_log_message("predicting")
  dataset["predicted"] = grid_results.predict(dataset[f"{x_col}"])

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

  # save information about each prediction
  df = pd.DataFrame({"Concordance": [concordance],
                     "CCCSMFA": [cccsmfa],
                     "Macro Recall": [macro_recall],
                     "Micro Recall": [micro_recall],
                     "Macro Precision": [macro_precision],
                     "Micro Precision": [micro_precision],
                     "Accuracy": [accuracy],
                     "best_model_params": [best_model_params]})
  print_log_message("writing dfs")
  df.to_csv(
      f"{testing_model_dir}/dataset_{dataset_num}_summary_stats.csv", index=False)


if __name__ == '__main__':

  best_model_dir = str(sys.argv[1])
  dataset_dir = str(sys.argv[2])
  testing_model_dir = str(sys.argv[3])
  best_model_params = str(sys.argv[4])
  int_cause = str(sys.argv[5])
  dataset_num = int(sys.argv[6])
  age_feature = str2bool(sys.argv[7])
  dem_feature = str2bool(sys.argv[8])

  main(best_model_dir, dataset_dir, testing_model_dir,
       best_model_params, int_cause, dataset_num, age_feature,
       dem_feature)
