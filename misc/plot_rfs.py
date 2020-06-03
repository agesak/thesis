"""compare mean ccc for number of trees/max depth rf grid search parameters"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cod_prep.claude.claude_io import makedirs_safely

DATE = "2020_05_03"
MCOD_DIR = "/ihme/cod/prep/mcod/process_data/"

def get_model_summaries(model_dir, param_string):

    direcs = []
    for root, dirs, files in os.walk(model_dir):
        for direc in dirs:
            if param_string in direc:
                direcs.append(direc)

    summaries = []
    for direc in direcs:
        fullpath = os.path.join(model_dir, direc)
        if os.path.exists(f"{fullpath}/summary_stats.csv"):
            summary = pd.read_csv(f"{fullpath}/summary_stats.csv")
        else:
            summary = pd.DataFrame()
        summaries.append(summary)

    if len(summaries) > 0:
        summaries = pd.concat(summaries)
    else:
        summaries = pd.DataFrame()
    return summaries


def plot_figure(df, int_cause, x_axis="max_depth", y_axis="mean_test_concordance"):
    plot = sns.FacetGrid(df, row="criterion", col="n_estimators")
    plot.map(plt.scatter, x_axis, y_axis)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    makedirs_safely(f"/home/j/temp/agesak/thesis/gridsearch_plots/{DATE}/")
    plt.savefig(f"/home/j/temp/agesak/thesis/gridsearch_plots/{DATE}/{int_cause}_rf_{DATE}.pdf")


# for int_cause in ["x59", "y34"]:
#     summaries = []
#     for tree in ["50", "60", "70", "80", "90", "150", "100", "200", "400", "600", "800", "1000", "1200", "1600", "2000", "2200"]:
#         for model_dir in [f"{thesis_dir}/{int_cause}/thesis/2020_04_27/rf", f"{thesis_dir}/{int_cause}/thesis/2020_04_16/rf"]:
#             summary = get_model_summaries(model_dir, f"model_{tree}_")
#             summaries.append(summary)

#     summaries = pd.concat(summaries)
#     summaries.columns = [x.replace("param_clf__estimator__", "")
#                          for x in list(summaries)]
#     plot_figure(summaries, int_cause)


for int_cause in ["x59", "y34"]:
    summaries = []
    for tree in ["60", "70", "80", "90", "100"]:
        summary = get_model_summaries(model_dir=f"{MCOD_DIR}/{int_cause}/thesis/{DATE}/rf", param_string=f"model_{tree}_")
        summaries.append(summary)
    summaries = pd.concat(summaries)
    summaries.columns = [x.replace("param_clf__estimator__", "")
                         for x in list(summaries)]
    plot_figure(summaries, int_cause)
