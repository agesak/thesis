import pandas as pd
import numpy as np
import os
import sys
import re

# read in best model results for each model type
# take these model to run on test datasets (so be able to generate 500 from dirichlet)
# generate evaluation metrics (precision, accuracy, ccc, cccsmfa on individual results?)


# do the dirichlet thing

# # read in and append all summary files for a given model type
# # pick model parameters with highest value for a given precision metric
# def get_best_fit(model_dir, short_model):
#     dfs = []
#     for root, dirs, files in os.walk(os.path.join(os.path.join(model_dir, short_model))):
#         for stats_dir in dirs:
#             df = pd.read_csv(os.path.join(
#                 model_dir, short_model, stats_dir, "summary_stats.csv"))
#             dfs.append(df)
#     df = pd.concat(dfs, sort=True, ignore_index=True)

#     # idk what ascending should be here bc it's negative?
#     best_fit = df.sort_values(by="mean_test_cccsfma",
#                               ascending=False).reset_index(drop=True).iloc[0:1]

#     # should i have been saving the model object?
#     # saves as pipeline object
#     return best_fit

# def generate_multiple_cause_rows(sample_df, test_df, cause):
#     """
#     Arguments:
#         sample_df: cause-specific df with number of rows equal to cause-specific proportion from dirichlet
#         test_df: true test df, corresponding to 25% of data
#         cause: cause of interest
#     Returns:
#         cause-specific df with chain cols randomly sampled from test df
#     """
#     multiple_cause_cols = [x for x in list(test_df) if "multiple_cause" in x]

#     # create multiple cause columns in the sample df
#     sample_df = pd.concat([sample_df, pd.DataFrame(columns=multiple_cause_cols)], sort=True, ignore_index=True)
#     # subset to only cause-specific rows in test df
#     cause_df = test_df.loc[test_df.cause_id==cause]
#     assert len(cause_df)!=0, "subsetting test df failed in creating 500 datasets"

#     # drop this column so .iloc will work
#     sample_df.drop(columns="cause_id", inplace=True)
#     # loop through rows of sample df
#     for index, row in sample_df.iterrows():
#         # should I be worried about replacement here?
#         # randomly sample 1 row in the cause-specific test df
#         chain = cause_df[multiple_cause_cols].sample(1).iloc[0]
#         # assign the multiple cause cols in the sample df to these chain cols
#         sample_df.iloc[[index],:] = chain.values
#     # add this column back
#     sample_df["cause_id"] = cause
#     return df

# def create_testing_datsets(test_df)

#     # dictionary of causes and their respective proportions in the data
#     cause_distribution = test_df['cause_id'].value_counts(normalize=True).to_dict()
#     # 500 dirichlet distributions based on test data cause distribution 
#     dts = np.random.dirichlet(alpha=list(cause_distribution.values()), size=500)

#     # such a random guess..
#     # should these be the length of the actual test df?
#     df_size = 1000

#     datasets = [np.NaN] * len(dts)
#     for i in range(0, len(dts)):
#         tdf = pd.DataFrame({"cause":[np.NaN] * df_size})
#         # dictionary of cause ids (order preserved i think?) to each dirichlet distribution
#         cd = dict(zip(cause_distribution.keys(), dts[i]))
#         df = []
#         for cause in cd.keys():
#             # proportion from dirichlet dictates how many rows are assigned to a given cause
#             s_tdf = tdf.sample(frac=cd[cause], replace=False).assign(cause_id=cause)
#             s_tdf = generate_multiple_cause_rows(s_tdf, test_df, cause)
#             df.append(s_tdf)
#         all_h = pd.concat(df, ignore_index=True, sort=True)
#         # compare
#         df['cause'].value_counts(normalize=True).to_dict()
#         # to - these fractions aren't alwayss the same 
#         cd
#         datasets[i] = all_h

#         all_h.to_csv()





def main(model_dir, short_model):

# this should be the model dir -> "/ihme/cod/prep/mcod/process_data/x59/thesis"
# create sample_dirichlet subfolder that also has model specific subfolders
# in each model specific subfolder, could have folders for "best" parameter type (and maybe date if consistent)
# this could be test_model_dir
    test_model_dir = f"{model_dir}/{short_model}/"

    # somehow be looping over model types or something?
    best_fit = get_best_fit(model_dir, short_model)

    test_df = pd.read_csv(f"{model_dir}/test_df.csv")




def run_pipeline(model, train_df, model_params, write_dir, int_cause):

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
    cccsfma_scorer = make_scorer(
        calculate_cccsmfa, greater_is_better=True)
    concordance_scorer = make_scorer(
        calculate_concordance, greater_is_better=True, int_cause=int_cause)

    scoring = {"precision": precision_scorer,
               "sensitivity": recall_scorer,
               "concordance": concordance_scorer,
               "cccsfma": cccsfma_scorer}

    gscv = GridSearchCV(pipeline, model_params, cv=5,
                        scoring=scoring, n_jobs=-1,
                        # just taking wild guesses here people
                        refit="cccsfma", verbose=6)

    grid_results = gscv.fit(train_df["cause_info"], train_df["cause_id"])

    print_log_message("saving model results")
    results = pd.DataFrame.from_dict(grid_results.cv_results_)
    # will change this name for each metric
    results.to_csv(f"{write_dir}/summary_stats.csv", index=False)


if __name__ == '__main__':

    model_dir = str(sys.argv[1])
    model = str(sys.argv[2])
    int_cause = str(sys.argv[3])
    short_model = str(sys.argv[4])

    main(model_params, model, model_dir, int_cause, short_model)



# r = re.compile("multiple_cause_[0-9]{1,2}$")
# multiple_cause_cols = list(filter(r.match, list(test_df)))
