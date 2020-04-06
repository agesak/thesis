import pandas as pd
import os







# read in best model results for each model type
# take these model to run on test datasets (so be able to generate 500 from dirichlet)
# generate evaluation metrics (precision, accuracy, ccc, cccsmfa on individual results?)



# do the dirichlet thing

test_df = pd.read_csv(f"{model_dir}/test_df.csv")




# read in and append all summary files for a given model type
# pick model parameters with highest value for a given precision metric
def main(model_dir, short_model):

    dfs = []
    for root, dirs, files in os.walk(os.path.join(os.path.join(model_dir, short_model))):
        for stats_dir in dirs:
            df = pd.read_csv(os.path.join(model_dir, short_model, stats_dir, "summary_stats.csv"))
            dfs.append(df)
    df = pd.concat(dfs, sort=True, ignore_index=True)

    # idk what ascending should be here bc it's negative?
    best_fit = df.sort_values(by="mean_test_cccsfma", ascending=False).reset_index(drop=True).iloc[0:1]

    # should i have been saving the model object?
    # saves as pipeline object
    return best_fit



if __name__ == '__main__':

    model_dir = str(sys.argv[1])
    model = str(sys.argv[2])
    model_dir = str(sys.argv[3])
    int_cause = str(sys.argv[4])
    short_model = str(sys.argv[5])

    main(model_params, model, model_dir, int_cause, short_model)