"""
07/04/2020: The N-code custom groups were incorrectly added to the mcause map
fixing that (and only that) here
"""
import re
import pandas as pd
from cod_prep.claude.configurator import Configurator

CONF = Configurator('standard')


def prep_all_inj_codes(code_system_id, mcod_map):
    '''
    Preps smaller ncode bins
    '''
    code_system_file_name_dict = {
        1: "Copy of 1-ICD10-mapping for X59 and Y34 -Jun-2019",
        6: "Copy of 2- -ICD9-mapping for X59 and Y34-Jun-2019"}

    filename = code_system_file_name_dict[code_system_id]
    df = pd.read_excel(
        f"/home/j/WORK/03_cod/01_database/mcod/injuries/maps_from_mohsen/{filename}.xlsx",
        sheet_name=1)

    df = df.loc[df["just for X59 and Y34"].str.contains(
        "^Extern|^NN|^Unspeci", flags=re.IGNORECASE, regex=True)]

    df.rename(columns={"icd_name": "cause_description",
                       "just for X59 and Y34": "package_description",
                       "yll_rdp": "package_name"}, inplace=True)

    code_system_type_dict = {1: "ICD10", 6: "ICD9"}
    df["code_system"] = code_system_type_dict[code_system_id]

    df["garbage_level"] = ""
    df["package_description"] = df["package_description"].str.lower()
    df = df[list(mcod_map)]

    return df


if __name__ == '__main__':

    mcod_map = pd.read_excel(
        "{}/mcause_map.xlsx".format(CONF.get_directory("process_inputs")))

    # archive the current map
    mcod_map.to_excel(
        f"{CONF.get_directory('process_inputs')}/_archive/mcause_map_2020_07_04.xlsx",
        index=False)

    # ger rid of current inj mapping
    mcod_map = mcod_map.loc[~mcod_map["package_description"].str.contains(
        "y34|x59|ncode|nn", flags=re.IGNORECASE, regex=True)]

    dfs = pd.DataFrame()

    for code_system_id in [1, 6]:
        df = prep_all_inj_codes(code_system_id, mcod_map)
        dfs = dfs.append(df, ignore_index=True)

    mcod_map = mcod_map.append([dfs], ignore_index=True)

    mcod_map.to_excel(
        f"{CONF.get_directory('process_inputs')}/mcause_map.xlsx",
        index=False)
