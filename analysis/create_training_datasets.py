
from cod_prep.downloaders import get_ages
from cod_prep.downloaders import add_age_metadata, add_cause_metadata, get_current_cause_hierarchy

from cod_prep.claude.configurator import Configurator


CONF = Configurator('standard')

# this?
df = df.loc[(df.age_group_id!=160) & (df.age_group_id!=283)]
# also make sure agg age groups

locs = get_location_metadata(gbd_round_id=6, location_set_id=35)

garbage_df = df.query(f"cause_id==743 & {int_cause}==1")
df = df.query(f"cause_id!=743 & {int_cause}!=1")

orig_cols = df.columns

keep_cols = DEM_COLS + ["cause_info", f"{int_cause}"] + [x for x in list(df) if "multiple_cause" in x]

injuries_restrictions = pd.read_csv("/homes/agesak/thesis/maps/injuries_overrides.csv")
injuries_restrictions = add_cause_metadata(injuries_restrictions, add_cols='cause_id', merge_col='acause',
                        cause_meta_df=cause_meta_df)
injuries_restrictions["age_start_group"] = injuries_restrictions["age_start_group"].fillna(0)

age_meta_df = get_ages(force_rerun=False, block_rerun=True)
cause_meta_df = get_current_cause_hierarchy(cause_set_id=4,
        **{'block_rerun': True, 'force_rerun': False})

df = add_age_metadata(
    df, add_cols=['age_group_years_start', 'age_group_years_end'],
    age_meta_df=age_meta_df
)

df = df.merge(injuries_restrictions, on='cause_id', how='left')

# age_group_years_end is weird, 0-14 means age_group_years_end 15
too_young = df["age_group_years_end"] <= df["age_start_group"]
too_old = df["age_group_years_start"] > df["age_end_group"]

df = df[~(too_young | too_old)]
df = df[orig_cols]


# creating test datasets may also need to factor in age somehow