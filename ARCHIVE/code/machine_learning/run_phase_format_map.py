"""Run formatting and mapping steps for mcod data.

Note: These steps are separated in the causes of death pipeline. They are combined here
into one step due to the limited use nature of many mcod datasets which cannot be saved
at the individual record level outside the L drive.
"""
from builtins import str
import sys
import pandas as pd
from importlib import import_module
from mcod_prep.mcod_mapping import MCoDMapper
from mcod_prep.utils.logging import ymd_timestamp, print_memory_timestamp
from cod_prep.claude.correct_restrictions import RestrictionsCorrector
from cod_prep.utils import print_log_message, report_duplicates
from cod_prep.downloaders import get_all_related_causes
from cod_prep.claude.configurator import Configurator
from cod_prep.claude.claude_io import write_phase_output, makedirs_safely

CONF = Configurator()
PROCESS_DIR = CONF.get_directory('mapping_process_data')
DIAG_DIR = PROCESS_DIR + '/{nid}/{extract_type_id}/{int_cause}'
ID_COLS = ['year_id', 'sex_id', 'age_group_id', 'location_id',
           'cause_id', 'code_id', 'nid', 'extract_type_id']


def get_drop_part2(int_cause, source):
    """Determine whether or not to drop part II of the death certificate.

    This is decided once we've reviewed the mapping of MCoD data and discussed
    coding practices and/or relevant disease etiology.
    """
    drop_p2_df = pd.read_csv(CONF.get_resource('drop_p2'), dtype={'drop_p2': bool})
    drop_p2_df = drop_p2_df[drop_p2_df['int_cause'] == int_cause]
    report_duplicates(drop_p2_df, 'int_cause')
    try:
        drop_p2 = drop_p2_df['drop_p2'].iloc[0]
    except IndexError:
        print(f"{int_cause} was not found in {CONF.get_resource('drop_p2')}")
    return drop_p2


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


def drop_non_mcause(df, int_cause, explore):
    """Drop rows where we cannot believe the death certificate.

    Mohsen decided in Oct. 2018 to exclude rows where there is only a single multiple cause
    of death and it matches the underlying cause. Also need to drop any rows where there
    are no causes in the chain; do this by ICD code.
    """
    chain_cols = [x for x in df.columns if ('multiple_cause_' in x)]
    df['num_chain_causes'] = 0
    for chain_col in chain_cols:
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
        # this used to save the output, but we don't always need that, so I took it out.
        # but someone could save this if they wanted to

    df = df.drop(['num_chain_causes', 'non_missing_chain'], axis=1)

    return df


def collect_cause_specific_diagnostics(df, acause_list):
    """Save diagnostic dataframe for certain causes."""
    if type(acause_list) != list:
        acause_list = [acause_list]
    df_list = []
    for acause in acause_list:
        diag_df = df.loc[df['cause_id'].isin(get_all_related_causes(acause))]
        df_list.append(diag_df)
    diag_df = pd.concat(df_list, ignore_index=True)
    return diag_df


def injuries_diagnostic(df):
    patterns = df["pattern"]
    df = df[(patterns.isin(patterns[patterns.duplicated()])) & (df.pattern != "")]
    return df


def get_id_value_cols(df, int_cause, data_type_id, inj_diag, explore, drop_p2):
    """Collapse data so it is not longer at individual record level."""
    group_cols = ID_COLS + [int_cause]

    if int_cause in ['x59', 'y34']:
        group_cols = ID_COLS + [int_cause, "pattern"]
        if inj_diag:
            group_cols += ["pII_ncodes", "pII_in_ncodes"]
    elif int_cause == 'infectious_syndrome':
        # can have more than one infectious syndrome, not mutually exclusive
        group_cols = ID_COLS + [x for x in df.columns if x.startswith('infectious_syndrome_')]

    # add on more columns for other options
    if explore:
        group_cols += ['drop_rows', 'cause_' + int_cause]
    if not drop_p2:
        group_cols += ['pII_' + int_cause]

    # make sure nothing was duplicated by accident
    group_cols = list(set(group_cols))

    # set value columns
    if data_type_id == 3:
        value_cols = ['admissions', 'deaths']
    else:
        value_cols = ['deaths']

    return group_cols, value_cols


def run_pipeline(year, source, int_cause, code_system_id, code_map_version_id,
                 cause_set_version_id, nid, extract_type_id, data_type_id,
                 diagnostic_acauses=None, explore=False, inj_diag=True):
    """Clean, map, and prep data for next steps."""
    drop_p2 = get_drop_part2(int_cause, source)

    print_log_message("Prepping data")
    formatting_method, args = get_formatting_method(source, data_type_id, year, drop_p2)
    df = formatting_method(*args)

    print_log_message("Dropping rows without multiple cause")
    df = drop_non_mcause(df, int_cause, explore)
    assert len(df) > 0, "No multiple cause data here!"

    # mapping from ICD code to code_id
    print_log_message("Mapping data")
    Mapper = MCoDMapper(int_cause, code_system_id, code_map_version_id, drop_p2)
    df = Mapper.get_computed_dataframe(df)


    # for saving diagnostic outputs
    outdir = DIAG_DIR.format(nid=nid, extract_type_id=extract_type_id, int_cause=int_cause)
    # underlying cause specific explorations
    if diagnostic_acauses is not None:
        diag_df = collect_cause_specific_diagnostics(df, diagnostic_acauses)
        makedirs_safely(outdir)
        diag_df.to_csv("{}/diagnostic_causes.csv".format(outdir), index=False)
    # save duplicated rows for injuries
    if inj_diag and (int_cause in ['x59', 'y34']):
        ddf = injuries_diagnostic(df)
        makedirs_safely(outdir)
        ddf.to_csv("{}/duplicated_pattern.csv".format(outdir), index=False)

    group_cols, value_cols = get_id_value_cols(df, int_cause, data_type_id,
                                               inj_diag, explore, drop_p2)
    print_memory_timestamp(df, f"Collapsing {value_cols} across {group_cols}")
    df = df.groupby(group_cols, as_index=False)[value_cols].sum()

    print_memory_timestamp(df, "Filtering cause-age-sex restrictions")
    Corrector = RestrictionsCorrector(
        code_system_id, cause_set_version_id, collect_diagnostics=False, verbose=True,
        groupby_cols=group_cols, value_cols=value_cols
    )
    df = Corrector.get_computed_dataframe(df)

    return df


def main(year, source, int_cause, code_system_id, code_map_version_id,
         cause_set_version_id, nid, extract_type_id, data_type_id):
    """Run pipeline."""
    df = run_pipeline(year, source, int_cause, code_system_id, code_map_version_id,
                      cause_set_version_id, nid, extract_type_id, data_type_id)
    print_log_message(f"Writing nid {nid}, extract_type_id {extract_type_id}")
    write_phase_output(df, "format_map", nid, extract_type_id, ymd_timestamp(), sub_dirs=int_cause)


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
