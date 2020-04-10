import pandas as pd
from db_queries import get_location_metadata
from mcod_prep.utils.mcause_io import get_mcause_data
from thesis_utils.modeling import read_in_data

LOCS = get_location_metadata(gbd_round_id=6, location_set_id=35)
BLOCK_RERUN = {'block_rerun': False, 'force_rerun': True}

# This included XX location-years of data from Mexico,
# XX location-years of data from Brazil, XX location-years of data
# from the United States, XX location-years of data from South Africa,
# XX location-years of data from Columbia,
# and XX location-years of data from Italy.
df = read_in_data(int_cause="x59")
df = df.merge(
    LOCS[["location_id", "parent_id", "level"]],
    on="location_id", how="left")
# map subnationals to country
df["country_id"] = np.where(
    df["level"] > 3, df["parent_id"],
    df["location_id"])
df.drop(columns=["parent_id", "level"], inplace=True)
df.rename(columns={"country_id": "location_id",
                   "location_id": "most_detailed_id"}, inplace=True)
# get country names
df = df.merge(
    LOCS[["location_id", "location_name"]],
    on="location_id", how="left")
ly_df = df.groupby("location_name", as_index=False).agg(
    {"most_detailed_id": "nunique", "year_id": "nunique"})
ly_df = ly_df.assign(
    location_years=ly_df["most_detailed_id"] * ly_df["year_id"])
ly_df.to_csv(
    "/home/j/temp/agesak/thesis/tables/location_years.csv", index=False)

# Data from XX, XX, and XX were ICD-10 coded,
# data from XX and XX were ICD-9 coded and
# data from XX contained both ICD-9 and ICD-10 coded deaths
df.groupby("location_name", as_index=False).agg(
    {"code_system_id": "unique"}).to_csv(
    "/home/j/temp/agesak/thesis/tables/icd_systems.csv", index=False)

# number of injuries related deaths - need total # deaths for each source
df.groupby("location_name", as_index=False).agg({"deaths":"sum"})
