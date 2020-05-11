import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cod_prep.claude.claude_io import makedirs_safely

DATE = "2020_05_03"
THESIS_DIR = "/ihme/cod/prep/mcod/process_data/"

def get_model_summaries(model_dir, param_string, maxdepth, nestimators):
    direcs = []
    for root, dirs, files in os.walk(model_dir):
        for direc in dirs:
            if param_string in direc:
                if (f"_{maxdepth}_" in direc) & (direc.endswith(f"_{nestimators}")):
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


def plot_figure(df, int_cause, x_axis="learning_rate", y_axis="mean_test_concordance", maxdepth=None, nestimators=None):
    plot = sns.FacetGrid(df, row="gamma", col="subsample")
    plot.map(plt.scatter, x_axis, y_axis)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    makedirs_safely(f"/home/j/temp/agesak/thesis/gridsearch_plots/{DATE}/")
    plt.savefig(f"/home/j/temp/agesak/thesis/gridsearch_plots/{DATE}/{int_cause}_xgb_nestimators_{nestimators}_maxdepth_{maxdepth}_{DATE}.pdf")


for int_cause in ["x59", "y34"]:
    maxdepth = 20
    nestimators = 80
    # order is learning rate, gamma, max depth, subsample
    summaries = []
    for learning_rate in [0.02,0.1,0.20,0.3,0.4]:
        # varying gamma, subsample,  max depth 100, n_estimators 80
        summary = get_model_summaries(
            f"{THESIS_DIR}/{int_cause}/thesis/{DATE}/xgb/", f"model_{learning_rate}", maxdepth=maxdepth, nestimators=nestimators)
        summaries.append(summary)

    summaries = pd.concat(summaries)
    summaries.columns = [x.replace("param_clf__estimator__", "")
                         for x in list(summaries)]
    summaries.rename(columns={"eta": "learning_rate"}, inplace=True)
    plot_figure(summaries, int_cause, maxdepth=maxdepth, nestimators=nestimators)
