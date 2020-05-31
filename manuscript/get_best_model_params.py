"""Get the best parameters for each classifier (Table 2)"""
from functools import reduce
import pandas as pd
import six

from cod_prep.utils.misc import print_log_message

date = "2020_05_23_most_detailed"
param_df = pd.read_csv("/homes/agesak/thesis/maps/parameters.csv")

for int_cause in ["x59", "y34"]:
    print_log_message(f"working on {int_cause}")
    dfs = []
    for short_name in ["nn", "rf", "multi_nb", "xgb"]:
        print_log_message(f"working on {short_name}")
        dataset_dir = f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{date}/{short_name}"
        # could pick any dataset here
        summaries = pd.read_csv(f"{dataset_dir}/dataset_1_summary_stats.csv")
        best_params = summaries.best_model_params.iloc[0]

        params = param_df[[x for x in list(param_df) if short_name in x]]
        params[f"{short_name}"] = params[f"{short_name}"].str.replace(
            "clf__estimator__", "")

        # format best params
        if isinstance(best_params, six.string_types):
            best_params = best_params.split("_")
        else:
            best_params = [best_params]

        param_kwargs = dict(zip(params.iloc[:, 0], best_params))
        model_params = pd.DataFrame.from_dict(param_kwargs, orient="index").reset_index()
        model_params.rename(columns={"index":"parameters", 0:f"{short_name}_values"}, inplace=True)
        dfs.append(model_params)
    df = reduce(lambda left, right: pd.merge(left, right, on=['parameters'],
                                             how='outer'), dfs)
    print_log_message(f"writing {int_cause} df")
    df.to_csv(f"/home/j/temp/agesak/thesis/model_results/{date}/{int_cause}_best_model_params.csv", index=False)


