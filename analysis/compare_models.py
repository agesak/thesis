from analysis.launch_models import ModelLauncher

model_dict = ModelLauncher.model_dict



# read in and append all summary files for a given model type
# pick summary file with highest value for a given precision metric
# final model: maybe average the measures for the precision metrics?


# read in best model results for each model type
# take these model to run on test datasets (so be able to generate 500 from dirichlet)
# generate evaluation metrics (precision, accuracy, ccc, cccsmfa on individual results?)