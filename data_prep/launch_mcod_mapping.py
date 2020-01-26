
"""Launch formatting, mapping, and redistribution steps for multiple cause data."""

import argparse
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from mcod_prep.utils.nids import get_datasets
from data_prep.mcod_mapping import MCoDMapper
from cod_prep.downloaders import (
    get_map_version, get_remove_decimal, print_log_message
)
from cod_prep.claude.configurator import Configurator
from cod_prep.claude.claude_io import delete_claude_output


class MCauseLauncher(object):

    conf = Configurator('standard')
    cache_options = {
        'force_rerun': True,
        'block_rerun': False,
        'cache_dir': "standard",
        'cache_results': True,
        'verbose': True

    }

    # USE NZL AND ITALY LINKAGE???
    source_memory_dict = {
        'TWN_MOH': '2G', 'MEX_INEGI': '7G', 'BRA_SIM': '12G', 'USA_NVSS': '20G',
        'COL_DANE': '2G', 'ITA_FVG_LINKAGE': '1G', 'NZL_LINKAGE_NMDS': '3G',
        'ZAF_STATSSA': '3G'}

    location_set_version_id = 420
    cause_set_version_id = 357
    thesis_code = "/homes/agesak/thesis/data_prep"

    def __init__(self, run_filters):
        self.run_filters = run_filters

    def prep_run_filters(self):
        datasets_kwargs = {'force_rerun': True, 'block_rerun': False}
        datasets_kwargs.update(
            {k: v for k, v in self.run_filters.items() if k not in [
                'intermediate_causes', 'phases']}
        )
        datasets = get_datasets(**datasets_kwargs)
        # drop External Causes of Death by Injuries and Poisonings source
        # these data are only used for drug overdoses (accidental poisoning)
        # and that's handled by some other code outside of this pipeline
        # removed Taiwan too
        datasets = datasets.loc[~(datasets['source'].isin(["EUROPE_INJ_POISON"]))]
        # datasets = datasets.loc[~(datasets['source'].isin(["EUROPE_INJ_POISON", "TWN_MOH"]))]
        datasets = datasets.drop_duplicates(
            ['nid', 'extract_type_id']).set_index(
            ['nid', 'extract_type_id'])[['year_id', 'code_system_id', 'source', 'data_type_id']]
        datasets['code_map_version_id'] = datasets['code_system_id'].apply(
            lambda x: get_map_version(x, 'YLL', 'best')
        )
        datasets['remove_decimal'] = datasets['code_system_id'].apply(
            lambda x: get_remove_decimal(x)
        )
        return datasets

    def launch_format_map(self, year, source, int_cause, code_system_id,
                          code_map_version_id, nid, extract_type_id, data_type_id):
        """Submit qsub for format_map phase."""
        delete_claude_output('format_map', nid, extract_type_id, sub_dirs=f"{int_cause}/thesis/")
        worker = f"{self.thesis_code}/run_phase_format_map.py"
        params = [int(year), source, int_cause, int(code_system_id), int(code_map_version_id),
                  int(self.cause_set_version_id), int(nid), int(extract_type_id), int(data_type_id)]
        jobname = f'format_map_{source}_{nid}_{year}_{int_cause}'
        try:
            memory = self.source_memory_dict[source]
        except KeyError:
            print(f"{source} is not in source_memory_dict. Trying with 5G.")
            memory = '5G'

        if data_type_id == 3:
            runtime = '02:00:00'
        else:
            runtime = '01::00'

        submit_mcod(
            jobname, 'python', worker, cores=1, memory=memory, params=params,
            verbose=True, logging=True, jdrive=True, runtime=runtime)

    def launch(self):
        datasets = self.prep_run_filters()

        for row in datasets.itertuples():
            nid, extract_type_id = row.Index
            for int_cause in self.run_filters['intermediate_causes']:
                print_log_message(f"launching jobs")
                self.launch_format_map(
                    row.year_id, row.source, int_cause, row.code_system_id,
                    row.code_map_version_id, nid, extract_type_id, row.data_type_id
                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="format and mapping for machine learning methods")
    parser.add_argument("intermediate_causes", help="intermediate cause(s) of interest",
                        nargs="+", choices=MCoDMapper.possible_int_causes)
    parser.add_argument('phases', help='data processing phases', nargs='+',
                        choices=['format_map'])
    parser.add_argument(
        'data_type_id', help="see cod.data_type for more options, you probably want 9", nargs='+')
    parser.add_argument('--source', nargs='*')
    parser.add_argument('--code_system_id', help="1 is ICD10, 6 is ICD9", nargs='*')
    parser.add_argument('--nid', nargs="*")
    args = parser.parse_args()
    print(vars(args))
    launcher = MCauseLauncher(vars(args))
    launcher.launch()
