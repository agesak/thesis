import pandas as pd
import numpy as np

from cod_prep.utils.misc import print_log_message
from thesis_utils.clf_switching import ClfSwitcher
from thesis_utils.model_evaluation import (calculate_cccsmfa,
                                           calculate_concordance)

from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV


def create_custom_scorers(int_cause):
    """
    """
    # https://stackoverflow.com/questions/32401493/how-to-create-customize-your-own-scorer-function-in-scikit-learn

    precision_scorer = make_scorer(
        precision_score, greater_is_better=True, average="micro")
    # my understanding is that this is the same as sensitivity
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    # tp / (tp + fn)
    recall_scorer = make_scorer(
        recall_score, greater_is_better=True, average="micro")
    cccsmfa_scorer = make_scorer(
        calculate_cccsmfa, greater_is_better=True)
    concordance_scorer = make_scorer(
        calculate_concordance, greater_is_better=True, int_cause=int_cause)

    return precision_scorer, recall_scorer, cccsmfa_scorer, concordance_scorer


def transform_measure_cols(df, measure, model_name, params):
    """Change """
    measure_dict = {"int": int, "float": float}
    measure_cols = df.loc[df[f"{model_name}_dtype"] ==
                          measure, f"{model_name}"].unique().tolist()
    for measure_col in measure_cols:
        params[measure_col] = [measure_dict[measure](params[measure_col])]

    return params, measure_cols


def format_gridsearch_params(model_name, param):

    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    measures = df[f"{model_name}_dtype"].dropna().unique().tolist()
    params = dict(zip(df[f"{model_name}"].unique().tolist(), param.split("_")))
    measure_cols = []
    for measure in measures:
        # dont need to do this for string
        if measure != "str":
            params, cols = transform_measure_cols(
                df, measure, model_name, params)
            measure_cols = measure_cols + cols
            params.update(params)

    # but all parameters must be lists
    str_cols = np.setdiff1d(
        df[f"{model_name}"].unique().tolist(), measure_cols).tolist()
    if not all(x == "nan" for x in str_cols):
        if "nan" in str_cols:
            str_cols.remove("nan")
        for col in str_cols:
            params[col] = [params[col]]
    return params


def run_pipeline(model, model_df, model_params, write_dir, int_cause):

    pipeline = Pipeline([
        ("bow", CountVectorizer(lowercase=False)),
        ("clf", ClfSwitcher())
    ])

    model_params.update({"clf__estimator": [eval(model)()]})

    precision_scorer, recall_scorer, cccsmfa_scorer, concordance_scorer = create_custom_scorers(
        int_cause)

    scoring = {"precision": precision_scorer,
               "sensitivity": recall_scorer,
               "concordance": concordance_scorer,
               "cccsmfa": cccsmfa_scorer}

    gscv = GridSearchCV(pipeline, model_params, cv=5,
                        scoring=scoring, n_jobs=3, pre_dispatch=6,
                        refit="cccsmfa", verbose=6)

    grid_results = gscv.fit(model_df["cause_info"], model_df["cause_id"])

    print_log_message("saving model results")
    results = pd.DataFrame.from_dict(grid_results.cv_results_)
    return results, grid_results
