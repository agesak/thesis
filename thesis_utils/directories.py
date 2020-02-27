import os
from cod_prep.utils import print_log_message


def get_limited_use_directory(source, int_cause):
    """Different input directories for limited use vs. non-limited use data."""
    limited_use = "/ihme/limited_use"
    thesis = f"mcod/{int_cause}"

    limited_use_paths = {
        "TWN_MOH": "LIMITED_USE/PROJECT_FOLDERS/GBD/TWN/VR/",
        "MEX_INEGI": "IDENT/PROJECT_FOLDERS/MEX/MULTIPLE_CAUSES_OF_DEATH_INEGI/",
        "BRA_SIM": "LIMITED_USE/PROJECT_FOLDERS/BRA/GBD_FROM_COLLABORATORS/SIM/",
        "USA_NVSS": "LIMITED_USE/PROJECT_FOLDERS/USA/NVSS_1989_2016_Y2019M02D27/1989_2016_CUSTOM_MORTALITY/"
    }
    if source in limited_use_paths.keys():
        limited_dir = os.path.join(limited_use, limited_use_paths[source], thesis)
    else:
        print_log_message(f"not using limited use directory for {source}")

    return limited_dir
