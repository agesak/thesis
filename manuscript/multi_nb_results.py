import pandas as pd
import numpy as np
from db_queries import get_location_metadata, get_cause_metadata
from cod_prep.downloaders import pretty_print, create_age_bins
from thesis_utils.misc import get_country_names


def format_classifier_results(int_cause, short_name):

    # just a test while I figure things out
    df = pd.read_csv(
        f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/2020_05_03/{short_name}/model_predictions.csv")

    df = get_country_names(df)
    df.drop(columns=["location_name", "Unnamed: 0", "cause_id"], inplace=True)
    df.rename(columns={"predictions": "cause_id"}, inplace=True)
    df = df.loc[(df.age_group_id != 283) & (df.age_group_id != 160)]
    df = create_age_bins(df, [39, 24, 224, 229, 47, 268, 294])
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
    rd = rd.groupby(['location_id', 'sex_id', 'year_id', 'age_group_id', 'cause_id'], as_index=False)['y34'].sum()
    rd["prop"] = rd.groupby(["age_group_id", "sex_id", "location_id", "year_id"], as_index=False)[
        f"{int_cause}"].transform(lambda x: x / float(x.sum(axis=0)))

    rd = pretty_print(rd)

    return rd



for int_cause in ["x59", "y34"]:
    for short_name in ["multi_nb"]:
        df = format_classifier_results(int_cause, short_name)
        rd = format_gbd_results(int_cause)
        rd.to_csv(
            f"/home/j/temp/agesak/thesis/{int_cause}_{short_name}_rd.csv", index=False)
        df.to_csv(
            f"/home/j/temp/agesak/thesis/{int_cause}_{short_name}_predictions.csv", index=False)
