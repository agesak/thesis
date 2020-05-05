import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


thesis_dir = "/ihme/cod/prep/mcod/process_data/"

def get_model_summaries(model_dir, param_string):

    direcs = []
    for root, dirs, files in os.walk(model_dir):
        for direc in dirs:
            if param_string in direc:
                # works for noww.. because max depth is 10 for all
                if re.search("_10_", direc):
                    direcs.append(direc)

    summaries = []
    for direc in direcs:
        fullpath = os.path.join(model_dir, direc)
        if os.path.exists(f"{fullpath}/summary_stats.csv"):
            summary = pd.read_csv(f"{fullpath}/summary_stats.csv")
        else:
            summary = pd.DataFrame()
        summaries.append(summary)

    summaries = pd.concat(summaries)

    return summaries


def plot_figure(df, int_cause, x_axis="learning_rate", y_axis="mean_test_concordance"):
    plot = sns.FacetGrid(df, row="gamma", col="subsample")
    plot.map(plt.scatter, x_axis, y_axis)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.savefig(f"/home/j/temp/agesak/thesis/gridsearch_plots/{int_cause}_xgb.pdf")


for int_cause in ["x59", "y34"]:

    # order is learning rate, gamma, max depth, subsample
    summaries = []
    for learning_rate in [0.02, 0.04, 0.06, 0.1, 0.14, 0.20, 0.25, 0.3]:
        # varying gamma, subsample,  max depth 10
        summary = get_model_summaries(
            f"{thesis_dir}/{int_cause}/thesis/2020_04_27/xgb/", f"model_{learning_rate}")
        summaries.append(summary)

    summaries = pd.concat(summaries)
    summaries.columns = [x.replace("param_clf__estimator__", "")
                         for x in list(summaries)]
    summaries.rename(columns={"eta": "learning_rate"}, inplace=True)
    plot_figure(summaries, int_cause)
