import pandas as pd
import numpy as np
from functools import reduce

from db_queries import get_location_metadata, get_cause_metadata
from cod_prep.downloaders import pretty_print, create_age_bins
from thesis_utils.misc import get_country_names

date = "2020_05_07"

def format_classifier_results(int_cause, short_name):

    # just a test while I figure things out
    df = pd.read_csv(
        f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{date}/{short_name}/model_predictions.csv")

    df = get_country_names(df)
    df.drop(columns=["location_name", "Unnamed: 0", "cause_id"], inplace=True)
    df.rename(columns={"predictions": "cause_id"}, inplace=True)
    df = df.groupby(["age_group_id", "sex_id", "location_id",
                     "year_id", "cause_id"], as_index=False)[f"{int_cause}"].sum()
    df["prop"] = df.groupby(["age_group_id", "sex_id", "location_id", "year_id"], as_index=False)[
        f"{int_cause}"].transform(lambda x: x / float(x.sum(axis=0)))
    df = pretty_print(df)

    return df


def format_gbd_results(int_cause):


    rd = pd.read_csv(
        f"/ihme/cod/prep/mcod/process_data/{int_cause}/rdp/2019_03_07/redistributed_deaths.csv")
    rd[[x for x in list(rd) if "inj" in x]] = rd[[
        x for x in list(rd) if "inj" in x]].fillna(0)
    rd = rd.groupby(['location_id', 'sex_id', 'year_id', 'age_group_id'],
                    as_index=False)[[x for x in list(rd) if "inj" in x]].sum()
    rd = pd.melt(rd, id_vars=['location_id', 'sex_id', 'year_id', 'age_group_id'], value_vars=[
                 x for x in list(rd) if "inj" in x], var_name="acause", value_name=int_cause)
    rd = rd.loc[rd[f"{int_cause}"] != 0]

    causes = get_cause_metadata(gbd_round_id=6, cause_set_id=3)
    injuries = causes.loc[(causes.acause.str.contains("inj"))
                          & (causes.most_detailed == 1)]
    inj_dict = injuries.set_index("acause")["cause_id"].to_dict()
    rd["cause_id"] = rd[["acause"]].apply(lambda x: x.map(inj_dict))

    restricted_targets = [729, 945]
    # should have been dropped last year (not most detailed/is yld only)
    restricted_targets += [704, 941]
    # x59 only unintentional
    if int_cause == "x59":
        restricted_targets += [721, 723, 725, 726, 727, 854, 941]
    rd = rd.loc[~(rd["cause_id"].isin(restricted_targets))]
    rd = get_country_names(rd)
    # make this right after dropping restricted targets
    rd = rd.groupby(['location_id', 'sex_id', 'year_id', 'age_group_id', 'cause_id'], as_index=False)[f"{int_cause}"].sum()
    rd["prop"] = rd.groupby(["age_group_id", "sex_id", "location_id", "year_id"], as_index=False)[
        f"{int_cause}"].transform(lambda x: x / float(x.sum(axis=0)))

    rd = pretty_print(rd)

    return rd

def choose_best_naive_bayes(int_cause):

    multi_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{date}/multi_nb/model_metrics_summary.csv")
    multi_df.rename(columns= lambda x: x + '_multi_nb' if x not in ['Evaluation metrics'] else x, inplace=True)

    complement_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{date}/complement_nb/model_metrics_summary.csv")
    complement_df.rename(columns= lambda x: x + '_complement_nb'  if x not in ['Evaluation metrics'] else x, inplace=True)

    bernoulli_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{date}/bernoulli_nb/model_metrics_summary.csv")
    bernoulli_df.rename(columns= lambda x: x + '_bernoulli_nb'  if x not in ['Evaluation metrics'] else x, inplace=True)

    df = reduce(lambda left,right: pd.merge(left,right,on=['Evaluation metrics'],
                                                how='outer'), [multi_df, complement_df, bernoulli_df])
    df.to_csv(f"/home/j/temp/agesak/thesis/model_results/test_set_summaries/{date}_{int_cause}_naivebayes_summary.csv", index=False)

    best_model = df[[x for x in list(df) if "Mean" in x]].idxmax(axis=1).iloc[0]

    return best_model


model_dict = {"x59":"", "y34":""}

for int_cause in ["x59", "y34"]:
    best_model = choose_best_naive_bayes(int_cause)
    best_model = best_model.replace("Mean_", "")
    model_dict.update({f"{int_cause}":best_model})


# will need to adapt this for other classifiers
for int_cause in ["x59", "y34"]:
    short_name = model_dict[int_cause]
    df = format_classifier_results(int_cause, short_name)
    rd = format_gbd_results(int_cause)
    rd.to_csv(
        f"/home/j/temp/agesak/thesis/model_results/{date}_{int_cause}_{short_name}_rd.csv", index=False)
    df.to_csv(
        f"/home/j/temp/agesak/thesis/model_results/{date}_{int_cause}_{short_name}_predictions.csv", index=False)
