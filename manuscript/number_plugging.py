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


"""RESULTS"""
x59 = pd.read_csv("/home/j/temp/agesak/thesis/model_results/2020_05_23_most_detailed/2020_05_23_most_detailed_x59_nn_predictions.csv")
y34 = pd.read_csv("/home/j/temp/agesak/thesis/model_results/2020_05_23_most_detailed/2020_05_23_most_detailed_y34_nn_predictions.csv")

# While the results from GBD 2019 had a large proportion of X59 deaths
# being classified as falls (xx%), the majority of deaths from the DNN 
# were redistributed to motor vehicle road injuries (XX%). 
df = x59.groupby("cause_name", as_index=False).agg(
    {"x59_deaths_thesis": "sum", "x59_deaths_GBD2019": "sum"})
df["percent_thesis"] = (df["x59_deaths_thesis"] / df["x59_deaths_thesis"].sum())*100
df["percent_GBD2019"] = (df["x59_deaths_GBD2019"] / df["x59_deaths_GBD2019"].sum())*100
df.sort_values("percent_GBD2019", ascending=False).iloc[0]
df.sort_values("percent_thesis", ascending=False).iloc[0]


# Likewise, the DNN predicted much higher proportions of X59 deaths 
# to be redistributed to pedestrian road injuries (xx%)
# and adverse effects of medical treatment (xx%),
# than was used in GBD 2019 (xx%, and xx% respectively).
df.loc[df.cause_name=="Pedestrian road injuries"].percent_thesis
df.loc[df.cause_name=="Adverse effects of medical treatment"].percent_thesis
df.loc[df.cause_name=="Pedestrian road injuries"].percent_GBD2019
df.loc[df.cause_name=="Adverse effects of medical treatment"].percent_GBD2019

# For the Y34 model, though physical violence by firearm received the largest 
# proportion of deaths in GBD 2019 (xx%), adverse effects of medical treatment 
# was predicted to receive the highest by the DNN (xx%) (Figure 4b). 
df = y34.groupby("cause_name", as_index=False).agg(
    {"y34_deaths_thesis": "sum", "y34_deaths_GBD2019": "sum"})
df["percent_thesis"] = (df["y34_deaths_thesis"] / df["y34_deaths_thesis"].sum())*100
df["percent_GBD2019"] = (df["y34_deaths_GBD2019"] / df["y34_deaths_GBD2019"].sum())*100
df.sort_values("percent_GBD2019", ascending=False).iloc[0]
df.sort_values("percent_thesis", ascending=False).iloc[0]

# Likewise, the DNN predicted a much higher proportion of deaths to be 
# redistributed to motor vehicle road injuries (xx%), and much lower 
# proportions for self-harm by other specified means (xx%), 
# physical violence by firearm (xx%), and falls (xx%)  than in GBD 2019 
# (xx%, xx%, xx%, and xx% respectively)(Figure 4b).
df.loc[df.cause_name=="Motor vehicle road injuries"].percent_thesis
df.loc[df.cause_name=="Self-harm by other specified means"].percent_thesis
df.loc[df.cause_name=="Physical violence by firearm"].percent_thesis
df.loc[df.cause_name=="Falls"].percent_thesis
df.loc[df.cause_name=="Motor vehicle road injuries"].percent_GBD2019
df.loc[df.cause_name=="Self-harm by other specified means"].percent_GBD2019
df.loc[df.cause_name=="Physical violence by firearm"].percent_GBD2019
df.loc[df.cause_name=="Falls"].percent_GBD2019

# Overall, the 5 causes with the highest proportion of redistributed
# X59 deaths were motor vehicle road injuries (xx%), pedestrian road 
# injuries (xx%), adverse effects of medical treatment (xx%), 
# falls (xx%), and motorcyclist road injuries (xx%) 
df = x59.groupby("cause_name", as_index=False).agg(
    {"x59_deaths_thesis": "sum"})
df["percent_thesis"] = (df["x59_deaths_thesis"] / df["x59_deaths_thesis"].sum())*100
df.sort_values(
    "percent_thesis", ascending=False)[["cause_name", "percent_thesis"]].head(5)

# For Y34, the top 5 causes were adverse effects of medical treatment 
# (xx%), motor vehicle road injuries (xx%), pedestrian road injuries 
# (xx%), self-harm by other specified means (xx%), motorcyclist road 
# injuries (xx%). 
df = y34.groupby("cause_name", as_index=False).agg(
    {"y34_deaths_thesis": "sum"})
df["percent_thesis"] = (df["y34_deaths_thesis"] / df["y34_deaths_thesis"].sum())*100
df.sort_values(
    "percent_thesis", ascending=False)[["cause_name", "percent_thesis"]].head(5)

# In looking by country for X59, the trend in motor vehicle road injuries
# was largely driven by the United States (xx% of US X59 deaths)
df = x59.groupby(["cause_name", "location_name"], as_index=False).agg(
    {"x59_deaths_thesis": "sum"}).query(
    "location_name=='United States of America'")
df["percent_thesis"] = (df["x59_deaths_thesis"] / df["x59_deaths_thesis"].sum())*100
df.loc[df.cause_name=="Motor vehicle road injuries"].percent_thesis
# while falls was driven largely by Mexico (xx% of Mexico X59 deaths) (Figure 5a). 
df = x59.groupby(["cause_name", "location_name"], as_index=False).agg(
    {"x59_deaths_thesis": "sum"}).query(
    "location_name=='Mexico'")
df["percent_thesis"] = (df["x59_deaths_thesis"] / df["x59_deaths_thesis"].sum())*100
df.loc[df.cause_name=="Falls"].percent_thesis

# For Y34, Italy drove the trend in adverse effects of medical treatment 
# (xx% of Italy Y34 deaths)
df = y34.groupby(["cause_name", "location_name"], as_index=False).agg(
    {"y34_deaths_thesis": "sum"}).query(
    "location_name=='Italy'")
df["percent_thesis"] = (df["y34_deaths_thesis"] / df["y34_deaths_thesis"].sum())*100
df.loc[df.cause_name=="Adverse effects of medical treatment"].percent_thesis
# while a large portion of the motorcyclist road injuries numbers were
# driven by Taiwan (xx% of Taiwan Y34 deaths) 
df = y34.groupby(["cause_name", "location_name"], as_index=False).agg(
    {"y34_deaths_thesis": "sum"}).query(
    "location_name=='Taiwan (Province of China)'")
df["percent_thesis"] = (df["y34_deaths_thesis"] / df["y34_deaths_thesis"].sum())*100
df.loc[df.cause_name=="Motorcyclist road injuries"].percent_thesis