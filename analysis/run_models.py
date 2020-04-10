import pandas as pd
import sys
from sklearn.externals import joblib

from cod_prep.utils.misc import print_log_message
from thesis_utils.grid_search import run_pipeline, format_gridsearch_params


def main(model_param, model_name, write_dir, train_dir, int_cause, short_name):

    model_df = pd.read_csv(f"{train_dir}/train_df.csv")
    print_log_message("formatting parameters")
    model_params = format_gridsearch_params(model_name, model_param)

    print_log_message("runninf pipeline")
    results, grid_results = run_pipeline(model_name, model_df, model_params,
                                         write_dir, int_cause)
    results.to_csv(f"{write_dir}/summary_stats.csv", index=False)
    joblib.dump(grid_results, f"{write_dir}/grid_results.pkl")


if __name__ == '__main__':

    write_dir = str(sys.argv[1])
    train_dir = str(sys.argv[2])
    model_param = str(sys.argv[3])
    model_name = str(sys.argv[4])
    short_name = str(sys.argv[5])
    int_cause = str(sys.argv[6])

    print(write_dir)
    print(train_dir)
    print(model_param)
    print(model_name)
    print(short_name)
    print(int_cause)
    main(model_param, model_name, write_dir, train_dir, int_cause, short_name)


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
