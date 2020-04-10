import pandas as pd
from mcod_prep.utils.mcause_io import get_mcause_data

BLOCK_RERUN = {'block_rerun': False, 'force_rerun': True}

df = get_mcause_data(
phase='format_map', sub_dirs="x59/thesis", source="BRA_SIM",
data_type_id=9,verbose=True, **BLOCK_RERUN)
# ex of getting loc-years of data for brazil
len(df.location_id.unique()) * len(df.year_id.unique())