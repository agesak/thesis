import pandas as pd
import numpy as np
from functools import reduce

from db_queries import get_location_metadata, get_cause_metadata
from cod_prep.downloaders import pretty_print, create_age_bins
from cod_prep.claude.claude_io import makedirs_safely
from cod_prep.utils.misc import print_log_message
from thesis_utils.misc import get_country_names

DATE = "2020_05_23_most_detailed"

def format_classifier_results(int_cause, short_name):
    model_dir = f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{DATE}"

    df = pd.read_csv(
        f"{model_dir}/{short_name}/model_predictions.csv")

    df = get_country_names(df)
    df.drop(columns=["location_name", "Unnamed: 0", "cause_id"], inplace=True)
    df.rename(columns={"predictions": "cause_id"}, inplace=True)
    df = df.groupby(["age_group_id", "sex_id", "location_id",
                     "year_id", "cause_id"], as_index=False)[f"{int_cause}"].sum()
    df["prop"] = df.groupby(["age_group_id", "sex_id", "location_id", "year_id"], as_index=False)[
        f"{int_cause}"].transform(lambda x: x / float(x.sum(axis=0)))

    df.rename(columns={"prop":"prop_thesis", f"{int_cause}":f"{int_cause}_deaths_thesis"})

    return df


def format_gbd_results(int_cause):

    rd = pd.read_csv(
        f"/ihme/cod/prep/mcod/process_data/{int_cause}/rdp/2019_03_07/redistributed_deaths.csv")
    rd[[x for x in list(rd) if "inj" in x]] = rd[[
        x for x in list(rd) if "inj" in x]].fillna(0)
    rd = rd.groupby(['location_id', 'sex_id', 'year_id', 'age_group_id'],
                    as_index=False)[[x for x in list(rd) if "inj" in x]].sum()
    rd = pd.melt(rd, id_vars=['location_id', 'sex_id', 'year_id', 'age_group_id'], value_vars=[
                 x for x in list(rd) if "inj" in x], var_name="acause", value_name=int_cause)
    rd = rd.loc[rd[f"{int_cause}"] != 0]

    causes = get_cause_metadata(gbd_round_id=6, cause_set_id=3)
    injuries = causes.loc[(causes.acause.str.contains("inj"))
                          & (causes.most_detailed == 1)]
    inj_dict = injuries.set_index("acause")["cause_id"].to_dict()
    rd["cause_id"] = rd[["acause"]].apply(lambda x: x.map(inj_dict))

    restricted_targets = [729, 945]
    # should have been dropped last year (not most detailed/is yld only)
    restricted_targets += [704, 941]
    # x59 only unintentional
    if int_cause == "x59":
        restricted_targets += [721, 723, 725, 726, 727, 854, 941]
    rd = rd.loc[~(rd["cause_id"].isin(restricted_targets))]
    rd = get_country_names(rd)
    # make this right after dropping restricted targets
    rd = rd.groupby(['location_id', 'sex_id', 'year_id', 'age_group_id', 'cause_id'], as_index=False)[f"{int_cause}"].sum()
    rd["prop"] = rd.groupby(["age_group_id", "sex_id", "location_id", "year_id"], as_index=False)[
        f"{int_cause}"].transform(lambda x: x / float(x.sum(axis=0)))

    return rd

def choose_best_naive_bayes(int_cause):
    # this will need to have a by age option

    multi_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{DATE}/multi_nb/model_metrics_summary.csv")
    multi_df.rename(columns= lambda x: x + '_multi_nb' if x not in ['Evaluation metrics'] else x, inplace=True)

    complement_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{DATE}/complement_nb/model_metrics_summary.csv")
    complement_df.rename(columns= lambda x: x + '_complement_nb'  if x not in ['Evaluation metrics'] else x, inplace=True)

    bernoulli_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{DATE}/bernoulli_nb/model_metrics_summary.csv")
    bernoulli_df.rename(columns= lambda x: x + '_bernoulli_nb'  if x not in ['Evaluation metrics'] else x, inplace=True)

    df = reduce(lambda left,right: pd.merge(left,right,on=['Evaluation metrics'],
                                                how='outer'), [multi_df, complement_df, bernoulli_df])

    makedirs_safely(f"/home/j/temp/agesak/thesis/model_results/{DATE}/")
    df.to_csv(f"/home/j/temp/agesak/thesis/model_results/{DATE}/{int_cause}_naivebayes_summary.csv", index=False)
    best_model = df[[x for x in list(df) if "Mean" in x]].idxmax(axis=1).iloc[0]

    return best_model


def choose_best_model(int_cause):

    bernoulli_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{DATE}/bernoulli_nb/model_metrics_summary.csv")
    bernoulli_df.rename(columns= lambda x: x + '_bernoulli_nb' if x not in ['Evaluation metrics'] else x, inplace=True)

    nn_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{DATE}/nn/model_metrics_summary.csv")
    nn_df.rename(columns= lambda x: x + '_nn'  if x not in ['Evaluation metrics'] else x, inplace=True)

    rf_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{DATE}/rf/model_metrics_summary.csv")
    rf_df.rename(columns= lambda x: x + '_rf'  if x not in ['Evaluation metrics'] else x, inplace=True)

    xgb_df = pd.read_csv(f"/ihme/cod/prep/mcod/process_data/{int_cause}/thesis/sample_dirichlet/{DATE}/xgb/model_metrics_summary.csv")
    xgb_df.rename(columns= lambda x: x + '_xgb'  if x not in ['Evaluation metrics'] else x, inplace=True)

    df = reduce(lambda left,right: pd.merge(left,right,on=['Evaluation metrics'],
                                                how='outer'), [bernoulli_df, nn_df, rf_df, xgb_df])

    makedirs_safely(f"/home/j/temp/agesak/thesis/model_results/{DATE}/")
    df.to_csv(f"/home/j/temp/agesak/thesis/model_results/{DATE}/{int_cause}_model_summary.csv", index=False)
    


# will only need to run this once ever tbh
# for int_cause in ["x59", "y34"]:
#     rd = format_gbd_results(int_cause)
#     rd = pretty_print(rd)
#     rd.to_csv(f"/home/j/temp/agesak/thesis/model_results/{int_cause}_gbd_2019.csv", index=False)


model_dict = {"x59":"", "y34":""}

# for int_cause in ["x59", "y34"]:
#     best_model = choose_best_naive_bayes(int_cause)
#     best_model = best_model.replace("Mean_", "")
#     model_dict.update({f"{int_cause}":best_model})

def update_model_dict(int_cause):
    best_model = choose_best_naive_bayes(int_cause)
    best_model = best_model.replace("Mean_", "")
    model_dict.update({f"{int_cause}":best_model})

    return model_dict

# for short_name in ["bernoulli_nb", "xgb", "rf", "nn"]:
#     print_log_message(f"working on {short_name}")
#     df = format_classifier_results(int_cause, short_name)
#     rd = format_gbd_results(int_cause)
# inconsistency here with short name for naive bayes
# here short name for all is "nb", because only the 
# best type of naive bayes will be used for final results
for int_cause in ["x59", "y34"]:
    print_log_message(f"working on {int_cause}")
    for short_name in ["rf", "nb", "xgb"]:
        print_log_message(f"working on {short_name}")
        if short_name == "nb":
            update_model_dict(int_cause)
            # get the short name associated with the best naive bayes model
            short_name = model_dict[int_cause]
        df = format_classifier_results(int_cause, short_name)
        rd = format_gbd_results(int_cause)
        rd.rename(columns={"prop":"prop_GBD2019", f"{int_cause}":f"{int_cause}_deaths_GBD2019"}, inplace=True)
        # merge on 2019 results
        # df = df.merge(rd, on=["age_group_id", "sex_id", "location_id", "year_id", "cause_id"], how="left")
        df = df.merge(rd, on=["age_group_id", "sex_id", "location_id", "year_id", "cause_id"], how="outer")
        df.rename(columns={"prop":"prop_thesis", f"{int_cause}":f"{int_cause}_deaths_thesis"}, inplace=True)
        df = pretty_print(df)
        df = df.fillna(0)
        makedirs_safely(f"/home/j/temp/agesak/thesis/model_results/{DATE}")
        df.to_csv(
            f"/home/j/temp/agesak/thesis/model_results/{DATE}/{DATE}_{int_cause}_{short_name}_predictions.csv", index=False)