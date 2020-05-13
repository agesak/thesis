import pandas as pd
import numpy as np

from cod_prep.utils.misc import print_log_message
from thesis_utils.clf_switching import ClfSwitcher
from thesis_utils.modeling import create_neural_network
from thesis_utils.model_evaluation import (calculate_cccsmfa,
                                           calculate_concordance)

from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              BaggingClassifier)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, recall_score,
                             accuracy_score, make_scorer)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from keras.wrappers.scikit_learn import KerasClassifier



def create_custom_scorers(int_cause):
    """
    """
    # https://stackoverflow.com/questions/32401493/how-to-create-customize-your-own-scorer-function-in-scikit-learn

    macro_precision_scorer = make_scorer(
        precision_score, greater_is_better=True, average="macro")
    micro_precision_scorer = make_scorer(
        precision_score, greater_is_better=True, average="micro")
    # my understanding is that this is the same as sensitivity
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    # tp / (tp + fn)
    macro_recall_scorer = make_scorer(
        recall_score, greater_is_better=True, average="macro")
    micro_recall_scorer = make_scorer(
        recall_score, greater_is_better=True, average="micro")
    accuracy_scorer = make_scorer(
        accuracy_score, greater_is_better=True)
    cccsmfa_scorer = make_scorer(
        calculate_cccsmfa, greater_is_better=True)
    concordance_scorer = make_scorer(
        calculate_concordance, greater_is_better=True, int_cause=int_cause)
    scorer_list = [macro_precision_scorer, micro_precision_scorer,
                   macro_recall_scorer, micro_recall_scorer,
                   accuracy_scorer, cccsmfa_scorer,
                   concordance_scorer]

    return scorer_list


def transform_measure_cols(df, measure, model_name, params):
    """Change """
    measure_dict = {"int": int, "float": float, "bool": bool}
    measure_cols = df.loc[df[
        f"{model_name}_dtype"] == measure, f"{model_name}"].unique().tolist()
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


def run_pipeline(model, short_name, model_df, model_params,
                 write_dir, int_cause, age_feature, dem_feature):

    if short_name == "svm_bag":
        model = {'model': BaggingClassifier,
                 'kwargs': {'base_estimator': eval(model)()},
                 'parameters': model_params}

        # create pipeline with bagging classifier
        pipeline = Pipeline([
            ("bow", CountVectorizer(lowercase=False)),
            ('name', model['model'](**model['kwargs']))
        ])

        cv_params = model['parameters']
    elif short_name == "nn":

        pipeline = Pipeline([
            ("bow", CountVectorizer(lowercase=False)),
            ("dense", FunctionTransformer(
                lambda x: x.todense(), accept_sparse=True)),
            ("clf", KerasClassifier(build_fn=create_neural_network,
                                    dropout_rate=0.2,
                                    output_nodes=len(
                                        model_df.cause_id.unique())))
        ])

        cv_params = model_params.copy()

    else:
        pipeline = Pipeline([
            ("bow", CountVectorizer(lowercase=False)),
            ("clf", ClfSwitcher())
        ])

        model_params.update({"clf__estimator": [eval(model)()]})
        cv_params = model_params.copy()

    scorer_list = create_custom_scorers(
        int_cause)

    scoring = {"macro_precision": scorer_list[0],
               "micro_precision": scorer_list[1],
               "macro_recall": scorer_list[2],
               "micro_recall": scorer_list[3],
               "accuracy": scorer_list[4],
               "cccsmfa": scorer_list[5],
               "concordance": scorer_list[6]}

    gscv = GridSearchCV(pipeline, cv_params, cv=5,
                        scoring=scoring, n_jobs=3, pre_dispatch=6,
                        refit="concordance", verbose=6)
    if age_feature:
        grid_results = gscv.fit(
            model_df["cause_age_info"], model_df["cause_id"])
    elif dem_feature:
        grid_results = gscv.fit(model_df["dem_info"], model_df["cause_id"])
    else:
        grid_results = gscv.fit(model_df["cause_info"], model_df["cause_id"])

    print_log_message("saving model results")
    results = pd.DataFrame.from_dict(grid_results.cv_results_)
    return results, grid_results
