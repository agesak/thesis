"""pull listing of packages so Mohsen can confirm which are injuries related"""

from cod_prep.downloaders import engine_room
import re

icd10 = engine_room.get_package_list(code_system_or_id=1, include_garbage_codes=True)
icd9 = engine_room.get_package_list(code_system_or_id=6, include_garbage_codes=True)

pd.concat([icd10, icd9]).drop_duplicates(["package_name", "package_description"])[["package_name", "package_description"]].to_csv("/homes/agesak/thesis.csv", index=False)