"""Launch formatting and mapping steps for multiple cause data."""

import argparse
import os
from mcod_prep.utils.mcod_cluster_tools import submit_mcod
from mcod_prep.utils.nids import get_datasets
from thesis_data_prep.mcod_mapping import MCoDMapper
from thesis_utils.directories import get_limited_use_directory
from thesis_utils.misc import str2bool
from cod_prep.downloaders import (
    get_map_version, get_remove_decimal, print_log_message
)
from cod_prep.claude.configurator import Configurator
from cod_prep.claude.claude_io import delete_claude_output, check_output_exists


class MCauseLauncher(object):

    conf = Configurator('standard')
    cache_options = {
        'force_rerun': True,
        'block_rerun': False,
        'cache_dir': "standard",
        'cache_results': True,
        'verbose': True

    }

    source_memory_dict = {
        'TWN_MOH': '2G', 'MEX_INEGI': '10G', 'BRA_SIM': '15G',
        'USA_NVSS': '20G', 'COL_DANE': '2G', 'ITA_ISTAT': '8G'}

    location_set_version_id = 420
    cause_set_version_id = conf.get_id('reporting_cause_set_version')
    thesis_code = "/homes/agesak/thesis/thesis_data_prep"
    limited_sources = ["TWN_MOH", "MEX_INEGI", "BRA_SIM", "USA_NVSS"]

    def __init__(self, run_filters):
        self.run_filters = run_filters

    def prep_run_filters(self):
        datasets_kwargs = {'force_rerun': True, 'block_rerun': False}
        datasets_kwargs.update(
            {k: v for k, v in self.run_filters.items() if k not in [
                'intermediate_causes', 'phase', 'inj_garbage']}
        )
        datasets = get_datasets(**datasets_kwargs)
        # GBD 2019: Drop Europe data because only for drug overdose
        # 5/23/2020: Drop South Africa because Mohsen said bad for injuries
        datasets = datasets.loc[~(
            datasets['source'].isin(["EUROPE_INJ_POISON", "ZAF_STATSSA"]))]
        datasets = datasets.drop_duplicates(
            ['nid', 'extract_type_id']).set_index(
            ['nid', 'extract_type_id'])[[
                'year_id', 'code_system_id', 'source', 'data_type_id']]
        datasets['code_map_version_id'] = datasets['code_system_id'].apply(
            lambda x: get_map_version(x, 'YLL', 'best')
        )
        datasets['remove_decimal'] = datasets['code_system_id'].apply(
            lambda x: get_remove_decimal(x)
        )
        return datasets

    def launch_format_map(self, year, source, int_cause, code_system_id,
                          code_map_version_id, nid, extract_type_id,
                          data_type_id):
        """Submit qsub for format_map phase."""
        # remove existing output
        if self.run_filters["inj_garbage"]:
            subdirs = f"{int_cause}/thesis/inj_garbage"
        else:
            subdirs = f"{int_cause}/thesis"
        delete_claude_output('format_map', nid, extract_type_id,
                             sub_dirs=subdirs)
        if source in self.limited_sources:
            limited_dir = get_limited_use_directory(
                source, int_cause, self.run_filters["inj_garbage"])
            if os.path.exists(
                    f"{limited_dir}/{nid}_{extract_type_id}_format_map.csv"):
                os.remove(
                    f"{limited_dir}/{nid}_{extract_type_id}_format_map.csv")
        worker = f"{self.thesis_code}/run_phase_format_map.py"
        params = [int(year), source, int_cause, int(code_system_id),
                  int(code_map_version_id), int(self.cause_set_version_id),
                  int(nid), int(extract_type_id), int(data_type_id),
                  self.run_filters["inj_garbage"]]
        if self.run_filters["inj_garbage"]:
            jobname = f'format_map_injgarbage_{source}_{nid}_{year}_{int_cause}'
        else:
            jobname = f'format_map_{source}_{nid}_{year}_{int_cause}'
        try:
            memory = self.source_memory_dict[source]
        except KeyError:
            print(f"{source} is not in source_memory_dict. Trying with 5G.")
            memory = '5G'

        if data_type_id == 3:
            runtime = '02:00:00'
        else:
            runtime = '06:00:00'

        submit_mcod(
            jobname, 'python', worker, cores=1, memory=memory, params=params,
            verbose=True, logging=True, jdrive=True, runtime=runtime)

    def launch_check_output(self, source, int_cause, nid,
                            extract_type_id, year_id, phase):
        if self.run_filters["inj_garbage"]:
            subdirs = f"{int_cause}/thesis/inj_garbage"
        else:
            subdirs = f"{int_cause}/thesis"
        if source in self.limited_sources:
            limited_dir = get_limited_use_directory(
                source, int_cause, self.run_filters["inj_garbage"])
            if not os.path.exists(
                    f"{limited_dir}/{nid}_{extract_type_id}_format_map.csv"):
                print_log_message(
                    f"no output found for {source} year {year_id} nid: {nid},"
                    f"extract_type_id: {extract_type_id}")
        else:
            if not check_output_exists(phase, nid, extract_type_id,
                                       sub_dirs=subdirs):
                print_log_message(
                    f"no output found for {source} year: {year_id} nid: {nid},"
                    f"extract_type_id: {extract_type_id}")

    def launch(self):
        datasets = self.prep_run_filters()

        if "format_map" in self.run_filters["phase"]:
            for row in datasets.itertuples():
                nid, extract_type_id = row.Index
                for int_cause in self.run_filters['intermediate_causes']:
                    print_log_message(f"launching jobs")
                    self.launch_format_map(
                        row.year_id, row.source, int_cause, row.code_system_id,
                        row.code_map_version_id, nid,
                        extract_type_id, row.data_type_id
                    )
        elif "check_output_exists" in self.run_filters["phase"]:
            for row in datasets.itertuples():
                nid, extract_type_id = row.Index
                for int_cause in self.run_filters['intermediate_causes']:
                    self.launch_check_output(row.source,
                                             int_cause, nid, extract_type_id,
                                             row.year_id, "format_map")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="format and mapping for machine learning methods")
    parser.add_argument("intermediate_causes",
                        help="intermediate cause(s) of interest",
                        nargs="+", choices=MCoDMapper.possible_int_causes)
    parser.add_argument('phase', help='data processing phases', nargs='+',
                        choices=['format_map', 'check_output_exists'])
    parser.add_argument(
        'data_type_id',
        help="see cod.data_type for more options, you probably want 9",
        nargs='+')
    parser.add_argument('--source', nargs='*')
    parser.add_argument('--year_id', type=int, nargs='*')
    parser.add_argument('--code_system_id',
                        help="1 is ICD10, 6 is ICD9", nargs='*')
    parser.add_argument('--nid', nargs="*")
    parser.add_argument("--inj_garbage",
                        help='format all injuries related garbage',
                        type=str2bool, nargs="?",
                        const=True, default=False)
    args = parser.parse_args()
    print(vars(args))
    launcher = MCauseLauncher(vars(args))
    launcher.launch()
