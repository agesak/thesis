from analysis.launch_models import ModelLauncher

model_dict = ModelLauncher.model_dict



# read in and append all summary files for a given model type
# pick summary file with highest value for a given precision metric
# final model: maybe average the measures for the precision metrics?


# read in best model results for each model type
# take these model to run on test datasets (so be able to generate 500 from dirichlet)
# generate evaluation metrics (precision, accuracy, ccc, cccsmfa on individual results?)


# def measure_prediction_quality(csmf_pred, y_test):
#     """Calculate population-level prediction quality (CSMF Accuracy)
    
#     Parameters
#     ----------
#     csmf_pred : pd.Series, predicted distribution of causes
#     y_test : array-like, labels for test dataset
    
#     Results
#     -------
#     csmf_acc : float
#     """
    
#     csmf_true = pd.Series(y_test).value_counts() / float(len(y_test))
#     temp = np.abs(csmf_true-csmf_pred)
#     csmf_acc = 1 - temp.sum()/(2*(1-np.min(csmf_true)))

#     return csmf_acc