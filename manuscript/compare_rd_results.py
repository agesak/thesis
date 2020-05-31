


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