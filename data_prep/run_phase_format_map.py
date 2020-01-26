"""Run formatting and mapping steps for mcod data.

Note: These steps are separated in the causes of death pipeline. They are combined here
into one step due to the limited use nature of many mcod datasets which cannot be saved
at the individual record level outside the L drive.
"""
import sys
import numpy as np
from importlib import import_module
from data_prep.mcod_mapping import MCoDMapper
from mcod_prep.utils.logging import ymd_timestamp
from mcod_prep.utils.causes import get_most_detailed_inj_causes
from cod_prep.utils import print_log_message
from cod_prep.claude.claude_io import write_phase_output

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
    try:
        formatting_method = getattr(
            import_module(f"mcod_prep.datasets.{clean_source}"), f"{clean_source}"
        )
    except AttributeError:
        print(f"No formatting method found! Check module & main function are named clean_{source}")
    return formatting_method, args


def drop_non_mcause(df, explore):
    """Drop rows where we cannot believe the death certificate.

    Mohsen decided in Oct. 2018 to exclude rows where there is only a single multiple cause
    of death and it matches the underlying cause. Also need to drop any rows where there
    are no causes in the chain; do this by ICD code.
    """
    chain_cols = [x for x in df.columns if ('multiple_cause_' in x)]
    df['num_chain_causes'] = 0
    for chain_col in chain_cols:
        print_log_message(chain_col)
        df.loc[df[chain_col] != '0000', 'num_chain_causes'] += 1

    # if there is only one chain, find the column where it is
    # in the US data, this is always the first chain column, but not the case for Mexico, e.g.
    df['non_missing_chain'] = chain_cols[0]
    for chain_col in chain_cols:
        df.loc[
            (df['num_chain_causes'] == 1) & (df[chain_col] != '0000'), 'non_missing_chain'
        ] = df[chain_col].copy()

    drop_rows = (
        ((df['num_chain_causes'] == 1) & (df['cause'] == df['non_missing_chain'])) |
        (df['num_chain_causes'] == 0)
    )

    if not explore:
        df = df[~drop_rows]
    else:
        df['drop_rows'] = 0
        df.loc[drop_rows, 'drop_rows'] = 1
        # this used to save the output, but we don't always need that, so I (Sarah) took it out.
        # but someone could save this if they wanted to

    df = df.drop(['num_chain_causes', 'non_missing_chain'], axis=1)

    return df


def run_pipeline(year, source, int_cause, code_system_id, code_map_version_id,
                 cause_set_version_id, nid, extract_type_id, data_type_id,
                 diagnostic_acauses=None, explore=False, drop_p2=False):
    """Clean, map, and prep data for next steps."""

    print_log_message("Formatting data")
    formatting_method, args = get_formatting_method(source, data_type_id, year, drop_p2=drop_p2)
    df = formatting_method(*args)

    print_log_message("Dropping rows without multiple cause")
    df = drop_non_mcause(df, explore)

    print_log_message("Mapping data")
    Mapper = MCoDMapper(int_cause, code_system_id, code_map_version_id, drop_p2=drop_p2)
    df = Mapper.get_computed_dataframe(df)

    # subset to rows where UCOD is injuries or any death is X59/y34
    df = df[[x for x in list(df) if not x.endswith(
        f"{int_cause}")] + [f"{int_cause}",
                            f"pII_{int_cause}", f"cause_{int_cause}"]]
    causes = get_most_detailed_inj_causes(int_cause, cause_set_id=4)
    df = df.loc[(df.cause_id.isin(causes)) | (df[f"{int_cause}"] == 1)]

    # some edits to cause_cols for BoW
    multiple_cause_cols = [x for x in list(df) if "cause" in x]
    multiple_cause_cols.remove("cause_id")
    df[multiple_cause_cols] = df[multiple_cause_cols].replace(
        "0000", np.NaN).replace("other", np.NaN)
    df["cause_info"] = df[[x for x in list(df) if "multiple_cause" in x]].fillna(
        "").astype(str).apply(lambda x: " ".join(x), axis=1)
    # df = df[["cause_id", "cause_info", f"{int_cause}", f"pII_{int_cause}"]]

    # skipping this for now because theres a groupby in here
    # print_memory_timestamp(df, "Filtering cause-age-sex restrictions")
    # Corrector = RestrictionsCorrector(
    #     code_system_id, cause_set_version_id, collect_diagnostics=False, verbose=True,
    #     groupby_cols=group_cols, value_cols=value_cols
    # )
    # df = Corrector.get_computed_dataframe(df)

    return df


def main(year, source, int_cause, code_system_id, code_map_version_id,
         cause_set_version_id, nid, extract_type_id, data_type_id):
    """Run pipeline."""
    df = run_pipeline(year, source, int_cause, code_system_id, code_map_version_id,
                      cause_set_version_id, nid, extract_type_id, data_type_id)
    print_log_message(f"Writing nid {nid}, extract_type_id {extract_type_id}")
    write_phase_output(df, "format_map", nid, extract_type_id, ymd_timestamp(),
                       sub_dirs=f"{int_cause}/thesis")


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

    main(year, source, int_cause, code_system_id, code_map_version_id,
         cause_set_version_id, nid, extract_type_id, data_type_id)
