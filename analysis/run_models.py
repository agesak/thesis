import pandas as pd
import numpy as np
import pickle
import sys
import ast

from cod_prep.utils.misc import print_log_message
from cod_prep.claude.claude_io import makedirs_safely
from thesis_utils.grid_search import ClfSwitcher

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV


"""to do
1. building pipeline to run multiple models
- later doing gridsearch
2. figuring out how to implement ccc on 500 test datasets
and whether or not this should be applied to all evaluation metrics
3. figure out how to implement the cccsfma - need function
"""


def format_params(model, param):

    df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")
    params = dict(zip(df[f"{model}"].unique().tolist(), param.split("_")))
    int_cols = df.loc[df[f"{model}_dtype"] ==
                      "int", f"{model}"].unique().tolist()
    # certain parameters must be integers
    for int_col in int_cols:
        params[int_col] = [int(params[int_col])]
    # but all parameters must be lists
    for col in np.setdiff1d(df[f"{model}"].unique().tolist(), int_cols):
        params[col] = [params[col]]
    return params


# precision, sensitivity, chance-corrected concordance (CCC)
# chance-corrected cause-specific mortality fraction (CCCSMF) accuracy
def run_pipeline(model, train_df, model_params, write_dir):

    pipeline = Pipeline([
        ("bow", CountVectorizer(lowercase=False)),
        ("clf", ClfSwitcher())
    ])

    model_params.update({"clf__estimator": [eval(model)()]})

    precision_scorer = make_scorer(
        precision_score, greater_is_better=True, average="micro")
    # my understanding is that this is the same as sensitivity
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    # tp / (tp + fn)
    recall_scorer = make_scorer(
        recall_score, greater_is_better=True, average="micro")

    scoring = {"precision": precision_scorer,
                "sensitivity": recall_scorer}

    gscv = GridSearchCV(pipeline, model_params, cv=5,
                        scoring=scoring, n_jobs=-1,
                        # just taking wild guesses here people
                        refit="precision", verbose=6)

    grid_results = gscv.fit(train_df["cause_info"], train_df["cause_id"])

    print_log_message("saving model results")
    results = pd.DataFrame.from_dict(grid_results.cv_results_)
    # will change this name for each metric
    results.to_csv(f"{write_dir}/summary_stats.csv", index=False)


def main(param, model, model_dir, int_cause, short_name):

    write_dir = f"{model_dir}/{short_name}/model_{param}"
    makedirs_safely(write_dir)

    train_df = pd.read_csv(f"{model_dir}/train_df.csv")
    model_params = format_params(model, param)

    run_pipeline(model, train_df, model_params, write_dir)




if __name__ == '__main__':

    model_params = str(sys.argv[1])
    model = str(sys.argv[2])
    model_dir = str(sys.argv[3])
    int_cause = str(sys.argv[4])
    short_model = str(sys.argv[5])

    main(model_params, model, model_dir, int_cause, short_model)


# pipeline = Pipeline([
#     ("bow", CountVectorizer(lowercase=False)),
#     ("clf", ClfSwitcher())
# ])

# # naive bayes, random forest, svm, GBT
# parameters = [
#     # {
#     #     "clf__estimator": [MultinomialNB()]
#     # },
#     {
#         # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
#         "clf__estimator": [RandomForestClassifier()],
#         "clf__estimator__n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
#         "clf__estimator__max_features": ["auto", "sqrt"],
#         "clf__estimator__max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
#         "clf__estimator__min_samples_split": [2, 5, 10],
#         "clf__estimator__min_samples_leaf": [1, 2, 4],
#         "clf__estimator__bootstrap": [True, False],
#         # "clf__estimator__n_jobs":[-1]
#     }
#     # {
#     #     "clf__estimator": [SVC()]
#     # },
#     #     {
#     #     "clf__estimator":[GradientBoostingClassifier()]
#     # }

# ]


# # chad thinks i should be using precision, accuracy, ccc, and ccscfma here (or the one that is most important which might be cccsfma?)
# custom_scorer = make_scorer(
#     precision_score, greater_is_better=True, average="micro")
# gscv = GridSearchCV(pipeline, parameters, cv=5,
#                     scoring=custom_scorer, n_jobs=-1, verbose=6)
# # this takes literal years..
# # i wonder if i could parallelize this by model and write the outputs somewhere...
# # it's SVC that takes forever... (and maybe random forest?)
# grid_results = gscv.fit(train_df["cause_info"], train_df["cause_id"])

# # then pick best model parameters for each classifier - other example might be better for this
# # want to output this as a table with metrics and model parameters..
# grid_result.best_params_
# grid_result.cv_results_

# # need to store these parameters as objects
# # should this be train_df?
# causes = test_df.cause_id.unique()
# probs = np.random.dirichlet([1] * len(causes), 500)
# # cause order for probabilities based off position
# cause_order = dict(zip(list(range(0, 18, 1)), list(causes)))


# # now dirichlet and resampling
# # idk why this doesnt work - lots of rows with na cause ids
# nrows = len(test_df)
# test1 = pd.DataFrame(
#     {"cause_id": nrows * [np.NaN], "cause_info": nrows * [np.NaN]})
# K = []
# for value in probs[0]:
#     index_val = list(probs[0]).index(value)
#     K = K + [value]
#     change = test1.sample(frac=value, replace=False).index
#     test1.loc[change, "cause_id"] = cause_order[index_val]

# K = []
# for value in probs[0]:
#     index_val = list(probs[0]).index(value)
#     K = K + [value]
#     ugh = test1.sample(frac=value, replace=False)
#     ugh["cause_id"] = cause_order[index_val]
#     test1.update(ugh)

# len(test1.loc[test1.cause_id.notnull()]) / len(test1)

# # sample to get cause info col too

# # same thing
# # cv = CountVectorizer(lowercase=False)
# # tf = cv.fit_transform(train_df["cause_info"])
# # clf = MultinomialNB().fit(tf, train_df["cause_id"])
# # new_counts = cv.transform(test_df["cause_info"])
# # predicted = clf.predict(new_counts)
# # test_df["predicted"] = predicted
# # np.mean(test_df.predicted == test_df.cause_id)


# # naive bayes, random forest, svm, GBT, Neural network
# # use Grid Search to choose between many parameters
# # text_clf = Pipeline([
# #     ("bow", CountVectorizer(lowercase=False)),
# #     ("naive_bayes", MultinomialNB())
# #     ])
# # text_clf.fit(train_df["cause_info"], train_df["cause_id"])
# # predicted = text_clf.predict(test_df["cause_info"])
# # np.mean(predicted == test_df["cause_id"])


# # from sklearn.base import BaseEstimator
# # class ClfSwitcher(BaseEstimator):
# # for building a pipeline
# # https://stackoverflow.com/questions/50285973/pipeline-multiple-classifiers - better way of viewing model results
# # http://www.davidsbatista.net/blog/2018/02/23/model_optimization/


# # Eval - precision, sensitivity, chance-corrected concordance, CCCSMFA
# # precision - TP/(TP+FP)
# precision_score(y_true=test_df.cause_id,
#                 y_pred=test_df.predicted, average="micro")

# # sensitivity (also known as recall) - TP/(TP+FN)
# # in a multi-class setting micro-averaged precision and recall are always the same.
# recall_score(y_true=test_df.cause_id,
#              y_pred=test_df.predicted, average="micro")

# # starting to sample from dirichlet
