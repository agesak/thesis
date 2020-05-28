"""Run formatting and mapping steps for mcod data."""
import sys
import numpy as np
import pandas as pd
from importlib import import_module
from thesis_data_prep.mcod_mapping import MCoDMapper
from thesis_data_prep.launch_mcod_mapping import MCauseLauncher
from thesis_utils.directories import get_limited_use_directory
from thesis_utils.misc import str2bool
from mcod_prep.utils.logging import ymd_timestamp
from mcod_prep.utils.causes import get_most_detailed_inj_causes
from cod_prep.utils import print_log_message, clean_icd_codes
from cod_prep.claude.claude_io import write_phase_output
from cod_prep.downloaders import engine_room

ID_COLS = ['year_id', 'sex_id', 'age_group_id', 'location_id',
           'cause_id', 'code_id', 'nid', 'extract_type_id']


def get_formatting_method(source, data_type_id, year, drop_p2):
    """Return the formatting method by source."""
    if data_type_id == 3:
        clean_source = 'clean_hospital_data'
        args = [source, year]
    else:
        clean_source = 'clean_' + source.lower()
        args = [year, drop_p2]
        # machine learning argument only for USA
        if source == "USA_NVSS":
            args += [True]
    try:
        formatting_method = getattr(
            import_module(f"mcod_prep.datasets.{clean_source}"),
            f"{clean_source}"
        )
    except AttributeError:
        print(
            f"No formatting method found! Check module & main function are named clean_{source}")
    return formatting_method, args


def drop_non_mcause(df, explore):
    """Drop rows where we cannot believe the death certificate.

    Mohsen decided in Oct. 2018 to exclude rows where
    there is only a single multiple cause
    of death and it matches the underlying cause.
    Also need to drop any rows where there
    are no causes in the chain; do this by ICD code.
    """
    chain_cols = [x for x in df.columns if ('multiple_cause_' in x)]
    df['num_chain_causes'] = 0
    for chain_col in chain_cols:
        print_log_message(chain_col)
        df.loc[df[chain_col] != '0000', 'num_chain_causes'] += 1

    # if there is only one chain, find the column where it is
    # in the US data, this is always the first chain column,
    # but not the case for Mexico, e.g.
    df['non_missing_chain'] = chain_cols[0]
    for chain_col in chain_cols:
        df.loc[
            (df['num_chain_causes'] == 1) & (
                df[chain_col] != '0000'), 'non_missing_chain'
        ] = df[chain_col].copy()

    drop_rows = (
        ((df['num_chain_causes'] == 1) & (
            df['cause'] == df['non_missing_chain'])) | (
            df['num_chain_causes'] == 0))

    if not explore:
        df = df[~drop_rows]
    else:
        df['drop_rows'] = 0
        df.loc[drop_rows, 'drop_rows'] = 1
        # this used to save the output, but we don't
        # always need that, so I (Sarah) took it out.
        # but someone could save this if they wanted to

    df = df.drop(['num_chain_causes', 'non_missing_chain'], axis=1)

    return df


def drop_duplicated_values(df, cols, fill):

    duplicated = df[cols].apply(
        pd.Series.duplicated, 1) & df[cols].astype(bool)
    dropdf = df.drop(columns=cols)
    df = pd.concat(
        [dropdf, df[cols].mask(duplicated, fill)], axis=1)

    return df


def format_for_bow(df, code_system_id):
    """Bag of words (bow) requires a single column with
    all ICD coded information concatenated together
    5/23/2020: Exploring the hierarchial nature of the ICD,
    adding the option to include less detailed ICD codes information in the bow
    Previous default, include: S00.01 (S0001 without decimal)
    Now have options to include: S00, and S
    Returns: df with with bow columns corresponding to
    1. most detailed icd codes only
    2. aggregated (3 digit) only code
    ICD 10 (because ICD 9 doesn't have letters):
    3. aggregated (3 digit) and letter
    4. most detailed and letter """
    multiple_cause_cols = [x for x in list(df) if "multiple_cause" in x]

    # first, drop any duplicated ICD codes by row
    df = drop_duplicated_values(df, multiple_cause_cols, fill="0000")

    # capture hierarchical nature of ICD codes
    feature_names = {1: "icd_letter",
                     3: "icd_aggregate_code", 4: "icd_one_decimal"}
    # ICD 9, care about 3 and 4 digit hierarchy
    # ICD 10, care about letter, aggregate code,
    # and detail past one decimal point
    digits = {6: [3, 4], 1: [1, 3, 4]}
    for col in multiple_cause_cols:
        print(col)
        # mostly for ICD 9
        df[col] = df[col].astype(str)
        for n in digits[code_system_id]:
            df[f"{feature_names[n]}_{col}"] = np.NaN
            df.loc[df[f"{col}"] != "0000",
                   f"{feature_names[n]}_{col}"] = df[col].apply(
                lambda x: x[0:n])

    df[multiple_cause_cols] = df[multiple_cause_cols].replace(
        "0000", np.NaN)

    # column with just most detailed ICD code information
    df["most_detailed_cause_info"] = df[multiple_cause_cols].fillna(
        "").astype(str).apply(lambda x: " ".join(x), axis=1)

    # just aggregate ICD code information
    df = drop_duplicated_values(df, [x for x in list(
        df) if "icd_aggregate" in x], fill=np.NaN)
    df["aggregate_only_cause_info"] = df[[x for x in list(
        df) if "icd_aggregate" in x]].fillna(
        "").astype(str).apply(lambda x: " ".join(x), axis=1)

    if code_system_id == 1:
        # aggregate ICD code information and letter
        df = drop_duplicated_values(df, [x for x in list(
            df) if "icd_letter" in x], fill=np.NaN)
        df["aggregate_and_letter_cause_info"] = df[[x for x in list(
            df) if ("icd_aggregate" in x) | ("icd_letter" in x)]].fillna(
            "").astype(str).apply(lambda x: " ".join(x), axis=1)

        # most detailed and letter
        df["most_detailed_and_letter_cause_info"] = df[[x for x in list(
            df) if "icd_letter" in x] + multiple_cause_cols].fillna(
            "").astype(str).apply(lambda x: " ".join(x), axis=1)
    else:
        # ICD 9 does not have a "letter", just retain former aspect of desired column
        df["aggregate_and_letter_cause_info"] = df["aggregate_only_cause_info"]
        df["most_detailed_and_letter_cause_info"] = df["most_detailed_cause_info"]

    df.drop(columns=[x for x in list(df) if "icd" in x], inplace=True)

    return df


def run_pipeline(year, source, int_cause, code_system_id, code_map_version_id,
                 cause_set_version_id, nid, extract_type_id, data_type_id,
                 inj_garbage, diagnostic_acauses=None,
                 explore=False, drop_p2=False):
    """Clean, map, and prep data for next steps."""

    print_log_message("Formatting data")
    formatting_method, args = get_formatting_method(
        source, data_type_id, year, drop_p2=drop_p2)
    df = formatting_method(*args)

    print_log_message("Dropping rows without multiple cause")
    df = drop_non_mcause(df, explore)

    print_log_message("Mapping data")
    Mapper = MCoDMapper(int_cause, code_system_id,
                        code_map_version_id, drop_p2=drop_p2)
    df = Mapper.get_computed_dataframe(df)

    cause_cols = [x for x in list(df) if ("cause" in x) & ~(
        x.endswith("code_original")) & ~(x.endswith(f"{int_cause}"))]
    cause_cols.remove("cause_id")
    # keep original "cause" information for
    # "cause" col is a string name in CoD cause map
    # after mapping to cause ids - (ex code id 103591)
    if source == "USA_NVSS":
        if code_system_id == 1:
            for col in cause_cols:
                df.loc[~(df[f"{col}"].str.match(
                    "(^[A-Z][0-9]{2,4}$)|(^0000$)")),
                    col] = df[f"{col}_code_original"]

    if inj_garbage:
        # FYI: This was a last minute addition to make plots of %X59/Y34
        # of injuries garbage for my manuscript
        # it's not needed for any analysis
        print_log_message(
            "subsetting to only rows with UCOD as injuries garbage codes")
        package_list = pd.read_excel(
            "/homes/agesak/thesis/maps/package_list.xlsx",
            sheet_name="mohsen_vetted")

        # get a list of all injuries garbage package names
        inj_packages = package_list.package_name.unique().tolist()

        # get the garbage codes associated with these garbage packages
        garbage_df = engine_room.get_package_list(
            code_system_or_id=code_system_id, include_garbage_codes=True)

        # subset df to only rows with injuries garbage as UCOD
        df = apply_garbage_map(df, garbage_df, inj_packages)
    else:
        # subset to rows where UCOD is injuries or any death is X59/y34
        df = df[[x for x in list(df) if not ((x.endswith(f"{int_cause}")) | (
            x.endswith("code_original")))] + [
            int_cause, f"pII_{int_cause}", f"cause_{int_cause}"]]

        causes = get_most_detailed_inj_causes(
            int_cause, cause_set_version_id=cause_set_version_id,
            **{'block_rerun': True, 'force_rerun': False})
        df = df.loc[(df.cause_id.isin(causes)) | (
            (df[f"{int_cause}"] == 1) & (df.cause_id == 743))]

        df = format_for_bow(df, code_system_id)
    df.drop(columns=[x for x in list(df) if "pII" in x], inplace=True)
    return df

def apply_garbage_map(df, g_df, inj_packages):
    """only keep rows with injuries garbage as UCOD"""

    g_df["garbage_code"] = clean_icd_codes(
        g_df["garbage_code"], remove_decimal=True)
    g_df = g_df.loc[g_df.package_name.isin(inj_packages)]
    garbage_codes = g_df.garbage_code.unique().tolist()
    df["keep"] = 0
    for n in reversed(range(2, 7)):
        df["keep"] = np.where(df.cause.isin(
            [x[0:n] for x in garbage_codes]), 1, df["keep"])

    df = df.query("keep==1")

    return df


def write_outputs(df, int_cause, source, nid, extract_type_id, inj_garbage):
    """
    write_phase_output - for nonlimited use data
    write to limited use folder - for limited use data"""

    if source in MCauseLauncher.limited_sources:
        limited_dir = get_limited_use_directory(source, int_cause, inj_garbage)
        print_log_message(f"writing {source} to limited use dir")
        print_log_message(limited_dir)
        df.to_csv(
            f"{limited_dir}/{nid}_{extract_type_id}_format_map.csv",
            index=False)
    else:
        if inj_garbage:
            print_log_message(
                "writing formatted df with only injuries garbage codes as UCOD"
            )
            subdirs = f"{int_cause}/thesis/inj_garbage"
        else:
            subdirs = f"{int_cause}/thesis"
        print_log_message(
            f"Writing nid {nid}, extract_type_id {extract_type_id}")
        write_phase_output(df, "format_map", nid, extract_type_id,
                           ymd_timestamp(), sub_dirs=subdirs)


def main(year, source, int_cause, code_system_id, code_map_version_id,
         cause_set_version_id, nid, extract_type_id, data_type_id,
         inj_garbage=False):
    """Run pipeline."""
    df = run_pipeline(year, source, int_cause, code_system_id,
                      code_map_version_id, cause_set_version_id,
                      nid, extract_type_id, data_type_id, inj_garbage)
    write_outputs(df, int_cause, source, nid, extract_type_id, inj_garbage)


if __name__ == '__main__':
    year = int(sys.argv[1])
    source = str(sys.argv[2])
    int_cause = str(sys.argv[3])
    code_system_id = int(sys.argv[4])
    code_map_version_id = int(sys.argv[5])
    cause_set_version_id = int(sys.argv[6])
    nid = int(sys.argv[7])
    extract_type_id = int(sys.argv[8])
    data_type_id = int(sys.argv[9])
    inj_garbage = str2bool(sys.argv[10])
    print(year)
    print(source)
    print(int_cause)
    print(code_system_id)
    print(code_map_version_id)
    print(cause_set_version_id)
    print(nid)
    print(extract_type_id)
    print(data_type_id)
    print(inj_garbage)
    print(type(inj_garbage))

    main(year, source, int_cause, code_system_id, code_map_version_id,
         cause_set_version_id, nid, extract_type_id, data_type_id, inj_garbage)
