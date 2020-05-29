# number plugging manuscript

import pandas as pd
import numpy as np
from db_queries import get_cause_metadata
from mcod_prep.utils.mcause_io import get_mcause_data
from thesis_utils.modeling import read_in_data
from thesis_utils.misc import get_country_names


# This included XX location-years of data from Mexico,
# XX location-years of data from Brazil, XX location-years of data
# from the United States, XX location-years of data from Columbia,
# and XX location-years of data from Italy.
df = read_in_data(int_cause="x59", code_system_id=None, inj_garbage=False)
df = get_country_names(df)
ly_df = df.groupby("location_name", as_index=False).agg(
    {"most_detailed_id": "nunique", "year_id": "nunique"})
ly_df = ly_df.assign(
    location_years=ly_df["most_detailed_id"] * ly_df["year_id"])
# ended up taking this out of table 1
ly_df.to_csv(
    "/home/j/temp/agesak/thesis/tables/location_years.csv", index=False)
# and replacing with this
df.groupby("location_name", as_index=False)["year_id"].agg(
    ["min", "max"]).reset_index().to_csv(
    "/home/j/temp/agesak/thesis/tables/year_range.csv", index=False)


# Data from XX, XX, and XX were ICD-10 coded,
# data from XX and XX were ICD-9 coded and
# data from XX contained both ICD-9 and ICD-10 coded deaths
df.groupby("location_name", as_index=False).agg(
    {"code_system_id": "unique"}).to_csv(
    "/home/j/temp/agesak/thesis/tables/icd_systems.csv", index=False)

# number of injuries related deaths - need total # deaths for each source
# df.groupby("location_name", as_index=False).agg({"deaths": "sum"})

# Deaths where an injuries-related ICD code was the
# underlying cause of death were mapped to one of XX
# most-detailed GBD injuries causes.
causes = get_cause_metadata(gbd_round_id=6, cause_set_id=3)
injuries = causes.loc[causes.acause.str.contains("inj")]
len(injuries.query("most_detailed==1"))

# Sentence: Of the XX million deaths available in these records,
# XX% were injuries related, with XX% of these injuries deaths being garbage coded.
# get just injuries related deaths.

# Part 1: Of the XX million deaths available in these records,
# could pick any int cause here
df = get_mcause_data(
    phase='format_map', sub_dirs="sepsis",
    source=["TWN_MOH", "MEX_INEGI", "BRA_SIM",
            "USA_NVSS", "COL_DANE", "ITA_ISTAT"],
    verbose=True, **{"force_rerun": True, "block_rerun": False})
# total deaths
total_deaths = df.deaths.sum()
df = get_country_names(df)
# total records from each study (Table 1)
df.groupby("location_name", as_index=False)["deaths"].sum(
).to_csv("/home/j/temp/agesak/thesis/tables/total_records.csv", index=False)

# Part 2: XX% were injuries related, with XX% of these injuries deaths being garbage coded.
# use y34 because includes intentional injuries
# first get number of gbd injuries coded deaths
y34 = read_in_data(int_cause="y34", code_system_id=None, inj_garbage=False)
y34 = y34[["location_id", "cause_id", "cause_y34", "y34", "deaths"]]
y34 = get_country_names(y34)
# remove y34 coded deaths
y34 = y34.loc[~((y34.cause_id == 743) & (y34["y34"] == 1))]
y34.groupby("location_name", as_index=False)["deaths"].sum().to_csv(
    "/home/j/temp/agesak/thesis/tables/gbd_injuries_records.csv", index=False)

# then get number of injuries garbage coded deaths - separated by whether or not y34 related (also can use for figure 2)
df = read_in_data(int_cause="y34", code_system_id=None, inj_garbage=True)
df = df[["location_id", "cause_id", "cause_y34", "y34", "deaths"]]
df = get_country_names(df)
df = df.groupby(["location_name", f"cause_y34"],
                as_index=False)["deaths"].sum()
df["percent_y34"] = df.groupby("location_name")[
    "deaths"].transform(lambda x: x / x.sum(axis=0))
df.to_csv(f"/home/j/temp/agesak/thesis/tables/percent_y34.csv", index=False)

# XX% were injuries related
df = pd.read_csv("/home/j/temp/agesak/thesis/tables/total_records.csv")
gbd = pd.read_csv("/home/j/temp/agesak/thesis/tables/gbd_injuries_records.csv")
garbage = pd.read_csv("/home/j/temp/agesak/thesis/tables/percent_y34.csv")
inj_related = ((gbd.deaths.sum() + garbage.deaths.sum()) /
               df.deaths.sum()) * 100

# with XX% of these injuries deaths being garbage coded.
percent_garbage = (garbage.deaths.sum() / gbd.deaths.sum()) * 100

# Part 3: Of the injuries garbage coded deaths XX% were X59
# and XX% were Y34, though this fraction varied greatly by country.
# now get x59 numbers (also can use for figure 2)
df = read_in_data(int_cause="x59", code_system_id=None, inj_garbage=True)
df = df[["location_id", "cause_id", "cause_x59", "x59", "deaths"]]
df = get_country_names(df)
df = df.groupby(["location_name", f"cause_x59"],
                as_index=False)["deaths"].sum()
df["percent_x59"] = df.groupby("location_name")[
    "deaths"].transform(lambda x: x / x.sum(axis=0))
df.to_csv(f"/home/j/temp/agesak/thesis/tables/percent_x59.csv", index=False)

# XX% were X59
x59 = pd.read_csv("/home/j/temp/agesak/thesis/tables/percent_x59.csv")
x59 = x59.groupby("cause_x59", as_index=False)["deaths"].sum()
x59["percent"] = x59["deaths"] / x59["deaths"].sum()
x59.iloc[1, 2]

# and XX% were Y34
y34 = pd.read_csv("/home/j/temp/agesak/thesis/tables/percent_y34.csv")
y34 = y34.groupby("cause_y34", as_index=False)["deaths"].sum()
y34["percent"] = y34["deaths"] / y34["deaths"].sum()
y34.iloc[0, 2]

# Table 1: number of injuries records - again use y34 because includes intentional injuries
gbd = pd.read_csv("/home/j/temp/agesak/thesis/tables/gbd_injuries_records.csv")
garbage = pd.read_csv("/home/j/temp/agesak/thesis/tables/percent_y34.csv")
gbd.rename(columns={"deaths": "gbd_injuries_deaths"}, inplace=True)
garbage = garbage.pivot(index="location_name", columns="cause_y34", values="deaths").reset_index().rename(
    columns={"external causes udi,type unspecified-y34": "y34_deaths", "other": "other_injuries_garbage_deaths"})
df = gbd.merge(garbage, on="location_name")
df["total_injuries_deaths"] = df[[
    x for x in list(df) if "deaths" in x]].sum(axis=1)
df.to_csv("/home/j/temp/agesak/thesis/tables/total_injuries_records.csv", index=False)
