import pandas as pd
import glob
import os
import numpy as np

from mcod_prep.utils.mcause_io import get_mcause_data
from cod_prep.utils.misc import print_log_message
from thesis_utils.directories import get_limited_use_directory
from thesis_data_prep.launch_mcod_mapping import MCauseLauncher
from thesis_data_prep.grid_search import ClfSwitcher

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV


BLOCK_RERUN = {'block_rerun': False, 'force_rerun': True}

"""to do
1. building pipeline to run multiple models
- later doing gridsearch
2. figuring out how to implement ccc on 500 test datasets
and whether or not this should be applied to all evaluation metrics
3. figure out how to implement the cccsfma - need function
"""

def read_in_data(int_cause):
    """Read in and append all MCoD data"""

    print_log_message("reading in not limited use data")
    # it's not good the sources are hard-coded
    udf = get_mcause_data(
        phase='format_map', source=['COL_DANE', 'ZAF_STATSSA'], sub_dirs=f"{int_cause}/thesis",
        data_type_id=9, assert_all_available=True,
        verbose=True, **BLOCK_RERUN)

    print_log_message("reading in limited use data")
    dfs = []
    for source in MCauseLauncher.limited_sources:
        limited_dir = get_limited_use_directory(source, int_cause)
        csvfiles = glob.glob(os.path.join(limited_dir, '*.csv'))
        for file in csvfiles:
            df = pd.read_csv(file)
            dfs.append(df)
    ldf = pd.concat(dfs, ignore_index=True, sort=True)
    all_df = pd.concat([udf, ldf], sort=True, ignore_index=True)
    # will only train/test where we know UCoD
    # see how final results change when subsetting to where x59==0 -
    # so basically filtering out rows where x59 in chain but ucod is gbd injuries cause
    df = all_df[["cause_id", "cause_info", f"{int_cause}"]].query("cause_id!=743")

    return df



df = read_in_data(int_cause)
# split train 75%, test 25%
train_df, test_df = train_test_split(df, test_size=0.25)


pipeline = Pipeline([
    ("bow", CountVectorizer(lowercase=False)),
    ("clf", ClfSwitcher())
])

# naive bayes, random forest, svm, GBT
parameters = [
    {
        "clf__estimator": [MultinomialNB()]
    },
    {
        "clf__estimator":[RandomForestClassifier()]
    },
    {
        "clf__estimator": [SVC()]
    },
        {
        "clf__estimator":[GradientBoostingClassifier()]
    }

]

# chad thinks i should be using precision, accuracy, ccc, and ccscfma here (or the one that is most important which might be cccsfma?)
custom_scorer = make_scorer(precision_score, greater_is_better=True,  average="micro")
gscv = GridSearchCV(pipeline, parameters, cv=5, scoring=custom_scorer, n_jobs=-1, verbose=6)
# this takes literal years..
# i wonder if i could parallelize this by model and write the outputs somewhere...
# it's SVC that takes forever... (and maybe random forest?)
grid_results = gscv.fit(train_df["cause_info"], train_df["cause_id"])

# then pick best model parameters for each classifier - other example might be better for this
grid_result.best_params_
grid_result.cv_results_

# need to store these parameters as objects
# should this be train_df?
causes = test_df.cause_id.unique()
probs = np.random.dirichlet([1] * len(causes), 500)
# cause order for probabilities based off position
cause_order = dict(zip(list(range(0, 18, 1)), list(causes)))


# now dirichlet and resampling
# idk why this doesnt work - lots of rows with na cause ids
nrows = len(test_df)
test1 = pd.DataFrame({"cause_id":nrows*[np.NaN], "cause_info":nrows*[np.NaN]})
K = []
for value in probs[0]:
    index_val = list(probs[0]).index(value)
    K = K + [value]
    change = test1.sample(frac=value, replace=False).index
    test1.loc[change,'cause_id'] = cause_order[index_val]

K = []
for value in probs[0]:
    index_val = list(probs[0]).index(value)
    K = K + [value]
    ugh = test1.sample(frac=value, replace=False)
    ugh["cause_id"] = cause_order[index_val]
    test1.update(ugh)

len(test1.loc[test1.cause_id.notnull()])/len(test1)

# sample to get cause info col too

# same thing
# cv = CountVectorizer(lowercase=False)
# tf = cv.fit_transform(train_df["cause_info"])
# clf = MultinomialNB().fit(tf, train_df["cause_id"])
# new_counts = cv.transform(test_df["cause_info"])
# predicted = clf.predict(new_counts)
# test_df["predicted"] = predicted
# np.mean(test_df.predicted == test_df.cause_id)



# naive bayes, random forest, svm, GBT, Neural network
# use Grid Search to choose between many parameters
# text_clf = Pipeline([
#     ('bow', CountVectorizer(lowercase=False)),
#     ('naive_bayes', MultinomialNB())
#     ])
# text_clf.fit(train_df["cause_info"], train_df["cause_id"])
# predicted = text_clf.predict(test_df["cause_info"])
# np.mean(predicted == test_df["cause_id"])



# from sklearn.base import BaseEstimator
# class ClfSwitcher(BaseEstimator):
# for building a pipeline 
# https://stackoverflow.com/questions/50285973/pipeline-multiple-classifiers - better way of viewing model results
# http://www.davidsbatista.net/blog/2018/02/23/model_optimization/




# Eval - precision, sensitivity, chance-corrected concordance, CCCSMFA
# precision - TP/(TP+FP)
precision_score(y_true=test_df.cause_id, y_pred=test_df.predicted, average="micro")

# sensitivity (also known as recall) - TP/(TP+FN)
# in a multi-class setting micro-averaged precision and recall are always the same.
recall_score(y_true=test_df.cause_id, y_pred=test_df.predicted, average="micro")

# starting to sample from dirichlet


