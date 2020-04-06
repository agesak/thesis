import pandas as pd
import numpy as np
import os
import sys

# read in best model results for each model type
# take these model to run on test datasets (so be able to generate 500 from dirichlet)
# generate evaluation metrics (precision, accuracy, ccc, cccsmfa on individual results?)


# do the dirichlet thing




test_df = pd.read_csv(f"{model_dir}/test_df.csv")
# dictionary of causes and their respective proportions in the data
cause_distribution = test_df['cause_id'].value_counts(normalize=True).to_dict()
# 500 dirichlet distributions based on test data cause distribution 
dts = np.random.dirichlet(alpha=list(cause_distribution.values()), size=500)

# such a random guess..
# should these be the length of the actual test df?
df_size = 1000

datasets = [np.NaN] * len(dts)
for i in range(0, len(dts)):
    tdf = pd.DataFrame({"cause":[np.NaN] * df_size})
    # dictionary of cause ids (order preserved i think?) to each dirichlet distribution
    cd = dict(zip(cause_distribution.keys(), dts[i]))
    df = []
    for cause in cd.keys():
        # proportion from dirichlet dictates how many rows are assigned to a given cause
        s_tdf = tdf.sample(frac=cd[cause], replace=False).assign(cause=cause)
        df.append(s_tdf)
        # compare
        s_tdf['cause'].value_counts(normalize=True).to_dict()
        # to - these fractions aren't always the same 
        cd
        # the next step would be to randomly sample rows in my real test df with each cause (to get multiple cause info)
    datasets[i] =  pd.concat(df, ignore_index=True, sort=True)




        # tdf["cause"] = tdf["cause"].sample(frac=temp[cause], replace=False).replace(np.NaN, 1)
        # tdf["cause"] = tdf.sample(frac=cd[cause], replace=False).assign(cause=cause).iloc[:,0]


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


# read in and append all summary files for a given model type
# pick model parameters with highest value for a given precision metric
def get_best_fit(model_dir, short_model):
    dfs = []
    for root, dirs, files in os.walk(os.path.join(os.path.join(model_dir, short_model))):
        for stats_dir in dirs:
            df = pd.read_csv(os.path.join(
                model_dir, short_model, stats_dir, "summary_stats.csv"))
            dfs.append(df)
    df = pd.concat(dfs, sort=True, ignore_index=True)

    # idk what ascending should be here bc it's negative?
    best_fit = df.sort_values(by="mean_test_cccsfma",
                              ascending=False).reset_index(drop=True).iloc[0:1]

    # should i have been saving the model object?
    # saves as pipeline object
    return best_fit


def main(model_dir, short_model):

    # somehow be looping over model types or something?
    best_fit = get_best_fit(model_dir, short_model)


if __name__ == '__main__':

    model_dir = str(sys.argv[1])
    model = str(sys.argv[2])
    model_dir = str(sys.argv[3])
    int_cause = str(sys.argv[4])
    short_model = str(sys.argv[5])

    main(model_params, model, model_dir, int_cause, short_model)
