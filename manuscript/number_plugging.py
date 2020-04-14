import pandas as pd
import numpy as np
from db_queries import get_location_metadata, get_cause_metadata
from mcod_prep.utils.mcause_io import get_mcause_data
from mcod_prep.utils.causes import get_most_detailed_inj_causes
from thesis_utils.modeling import read_in_data

LOCS = get_location_metadata(gbd_round_id=6, location_set_id=35)


def get_country_names(df):
    """Map all subnationals to country-level
    Arguments:
            df: df with location_id column with some subnational level rows
    Returns:
        df - most_detailed_id column with most-detailed location id for a given GBD location
           - location_column with only country-level location names for all location ids
    """
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
    return df


# This included XX location-years of data from Mexico,
# XX location-years of data from Brazil, XX location-years of data
# from the United States, XX location-years of data from South Africa,
# XX location-years of data from Columbia,
# and XX location-years of data from Italy.
df = read_in_data(int_cause="x59")
df = get_country_names(df)
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
df.groupby("location_name", as_index=False).agg({"deaths": "sum"})

# Of the XX million deaths available in these records,
# injuries deaths make up XX%, with XX% of these injuries
# deaths coded as X59 and XX% coded as Y34, respectively.
# could pick any int cause here
df = get_mcause_data(
    phase='format_map', sub_dirs="sepsis",
    source=["TWN_MOH", "MEX_INEGI", "BRA_SIM", "USA_NVSS",
            "COL_DANE", "ZAF_STATSSA", "ITA_ISTAT"],
    verbose=True, **{"force_rerun": True, "block_rerun": False})
# total deaths
total_deaths = df.deaths.sum()
df = get_country_names(df)
# total records from each study (Table 1)
df.groupby("location_name", as_index=False)["deaths"].sum(
).to_csv("/home/j/temp/agesak/thesis/tables/total_records.csv", index=False)

y34 = read_in_data(int_cause="y34")
injuries = y34.deaths.sum()
# percent injuries
(injuries / total_deaths) * 100
# percent y34
(len(y34.query("y34==1")) / injuries) * 100
# number injuries records by country
y34 = get_country_names(y34)
y34.groupby("location_name", as_index=False)["deaths"].sum().to_csv(
    "/home/j/temp/agesak/thesis/tables/injuries_records.csv", index=False)
# percent x59
x59 = read_in_data(int_cause="x59")
(len(x59.query("x59==1")) / injuries) * 100

# Deaths where an injuries-related ICD code was the
# underlying cause of death were mapped to one of XX
# most-detailed GBD injuries causes.
causes = get_cause_metadata(gbd_round_id=6, cause_set_id=3)
injuries = causes.loc[causes.acause.str.contains("inj")]
len(injuries.query("most_detailed==1"))
