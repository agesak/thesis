"""table of mean ccc for xgb/rf classifiers across 500 test datasets"""
import pandas as pd
import os

date = "2020_05_07"
int_cause = "x59"
model_dir = f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/{date}"

all_df = []
for short_name in ["xgb", "rf"]:
    dfs = []
    for root, dirs, files in os.walk(os.path.join(
            os.path.join(model_dir, short_name))):
        for stats_dir in dirs:
            if os.path.exists(os.path.join(
                model_dir, short_name, stats_dir, "summary_stats.csv")):
                df = pd.read_csv(os.path.join(
                    model_dir, short_name, stats_dir, "summary_stats.csv"))
                df["model_params"] = stats_dir
            else:
                df = pd.DataFrame()
                df["model_params"] = stats_dir
            dfs.append(df)
    df = pd.concat(dfs, sort=True, ignore_index=True)
    df = df.sort_values(by="mean_test_concordance",
                ascending=False)[["model_params" ,"mean_test_concordance"]]
    df["short_name"] = short_name
    all_df.append(df)
alls = pd.concat(all_df, sort=True, ignore_index=True)

alls.to_csv(f"/home/j/temp/agesak/thesis/model_results/{date}/{date}_{int_cause}_xgb_rf_summary.csv", index=False)