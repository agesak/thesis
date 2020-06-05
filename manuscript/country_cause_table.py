"""plots by cause for appendix (table 3)"""

import pandas as pd
from cod_prep.claude.claude_io import makedirs_safely

DATE = "2020_05_23_most_detailed"

for int_cause in ["x59", "y34"]:

    df = pd.read_csv(
        f"/home/j/temp/agesak/thesis/model_results/{DATE}/{DATE}_{int_cause}_nn_predictions.csv")

    df = df.groupby("cause_name", as_index=False).agg(
    {f"{int_cause}_deaths_thesis": "sum", f"{int_cause}_deaths_GBD2019": "sum"})
    df[f"{int_cause}_deaths_GBD2019"] = df[f"{int_cause}_deaths_GBD2019"].round().astype(int)
    df["DNN by cause proportion"] = (df[f"{int_cause}_deaths_thesis"] / df[f"{int_cause}_deaths_thesis"].sum())*100
    df["GBD2019 by cause proportion"] = (df[f"{int_cause}_deaths_GBD2019"] / df[f"{int_cause}_deaths_GBD2019"].sum())*100
    df["DNN by cause proportion"] = df["DNN by cause proportion"].round(3).astype(str) + "%"
    df["GBD2019 by cause proportion"] = df["GBD2019 by cause proportion"].round(3).astype(str) + "%"
    df.rename(columns={"cause_name": "Cause Name", f"{int_cause}_deaths_thesis": f"{int_cause.upper()} DNN Deaths",
                      f"{int_cause}_deaths_GBD2019": f"{int_cause.upper()} GBD2019 Deaths"}, inplace=True)
    makedirs_safely(f"/home/j/temp/agesak/thesis/tables/{DATE}/")
    df.to_csv(
        f"/home/j/temp/agesak/thesis/tables/{DATE}/{int_cause}_cause_table.csv", index=False)
