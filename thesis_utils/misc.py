import argparse
import numpy as np
import os

from db_queries import get_location_metadata


LOCS = get_location_metadata(gbd_round_id=6, location_set_id=35)


def chunks(list_arg, n):
    n = max(1, n)
    return (list_arg[i:i + n] for i in range(0, len(list_arg), n))


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def remove_if_output_exists(file_dir, file):

    filepath = os.path.join(file_dir, file)
    if os.path.exists(filepath):
        os.unlink(filepath)


def get_country_names(df):
    """Map all subnationals to country-level
    Arguments:
            df: df with location_id column with some subnational level rows
    Returns:
        df - most_detailed_id column with most-detailed location id for a given GBD location
           - location_name column with only country-level location names for all location ids
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
