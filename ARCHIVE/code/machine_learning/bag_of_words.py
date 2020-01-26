
import pandas as pd
from mcod_prep.utils.mcause_io import get_mcause_data
from importlib import import_module


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


formatting_method, args = get_formatting_method(source="TWN_MOH", data_type_id=9, year=2008, drop_p2=True)
df = formatting_method(*args)