# Machine Learning for Injuries Cause of Death Assignment: A New Method for the Global Burden of Disease Study
### A master's thesis completed for the University of Washington Master of Public Health Program (2020)
### Thesis committee: Dr. Mohsen Naghavi and Dr. Abraham Flaxman
### Kareha Agesa

## Overview
This repository contains code for my 2020 master's thesis exploring machine learning for X59 and Y34 garbage code redistribution. The code is structured as follows:
```
.
├── analysis            # pipeline to launch classifiers and perform model evaluation
├── manuscript          # figures/tables/number plugging for manuscipt
├── maps                # excel files pertaining to attributes of models
├── misc                # misc scripts to vet model outputs
├── thesis_data_prep    # formatting and mapping raw input data
├── thesis_utils        # various functions essential to pipeline
├── README.md 
├── environment.yml 
```

## Getting Starting
To obtain the python environment used in this analysis run\*:
```bash
conda env create -f environment.yml
```

## _analysis_
### Overview
The actions carried out in the analysis folder (creation of the train/test datasets, model fitting, and model evaluation) are launched using the `ModelLauncher` class in `launch_models.py`. This class accepts command line arguments related to which classifier to run, which intermediate cause (either X59 or Y34) to use, which phase of the pipeline (defined below), and various characteristics of the data/model. This folder is structured as follows:

```
.
├── create_test_datasets.py         # worker script to generate each of 500 test datasets from a Dirichlet distribution
├── launch_models.py                # main class that creates train/test datasets, launches each classifier, and performs model evaluation
├── run_models.py                   # fits each classifier on the training data
├── run_testing_predictions.py      # performs predictions on each of the 500 generated test datasets using each classifier
├── run_unobserved_predictions.py   # creates file of summary statistics across all evaluation metrics for the 500 generated test datasets, refits a given classifier on all unobserved data (train and test), predicts on unobserved data

```
### _Model Phases_
Each phase of this pipeline is defined as follows:
- train_test: creation of the (75%) train and (25%) test datasets
- create_test_datasets: creation of the 500 generated test datasets used for model evaluation
- launch_training_model: fits a given classifier on the training data (and performs 5 fold cross validation using a [gridsearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html))
- launch_testing_models: predict on each of the 500 generated test datasets for a given classifier
- launch_int_cause_predictions: refit on all observed data and predict on the unobserved data

\* The order of these phases is not defined in the code, but you should start by creating the train/test datasets, then if you like you could either launch the training models or create the test datasets. Launching the testing models must come after the 500 generated test datasets were created in phase: create_test_datasets, then you can launch_int_cause_predictions.

### _Machine Learning Implementations_
1. [Bag of words](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
2. [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) 
2. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. [Gradient Boosted Trees](https://xgboost.readthedocs.io/en/latest/python/index.html)
4. [Deep Neural Network](https://keras.io/)

### _Various Data/Model Attributes_
The `ModelLauncher` class offers various flexibility regarding the input data and model attributes. For the input data, you have the ability to model at either the most detailed or the country location level and the ability to subset to only data for a given ICD system (either ICD 9 or ICD 10). The class also offers flexibility in selecting various attributes of the data as features in the bag of words. For example (keeping standard the inclusion of ICD codes as features), models can additionally be run with just age as a feature or age, sex, location, and year as features. The ability also exists to run separate models by age. As a supplementary analysis, the hierarchial nature of the ICD was explored as features in the bag of words. For example, instead of just including the most detailed ICD code given in the data, models were tried with just 3 digit ICD codes, 3 digit ICD codes and the ICD letter (for ICD 10), and most detailed code and the letter (for ICD 10). These are denoted as "most_detailed", "aggregate_only", "aggregate_and_letter", and "most_detailed_and_letter". Additionally in GBD 2019, Mohsen collapsed all N-codes into custom groups (as a way to ease modeling), so there exists an option to use these groups as features in the bag of words (instead of the ICD codes themselves). Denoted "grouped_ncode" in the `ModelLauncher`.


## _manuscript_
### Overview
Various files related to generating figures and tables for my thesis manuscipt, along with a catalogue of how numbers were generated in the main text.
```
.
├── by_country.R                # creates bar graphs of the fraction of x59/y34 deaths redistributed to top 5 causes with highest proportion of redistributed deaths by country
├── compare_rd_results.py       # creates csvs with redistribution numbers and fractions for 1. GBD 2019 and 2. for each classifier by age, sex, location (country), and year
├── country_cause_table.py      # creates csv of by cause redistribution proportions and numbers for X59/Y34 and the best classifier (DNN)
├── get_best_model_params.py    # creates csv of best model parameters for X59/Y34 for each classifier
├── number_plugging.py          # record of how all numbers were calculated in the manuscript
├── percent_garbage.py          # creates bar graph of the percent of all injuries garbage that is X59/Y34 by country
├── plot_rd_props.R             # creates by cause bar plots of redistribution proportions
```

## _maps_
### Overview
Some helpful csvs/excel files. 
```
.
├── injuries_overrrides.csv     # GBD injuries age/sex restrictions applied to the input data prior to modeling
├── package_list.xlsx           # denotes which garbage packages are injuries-related (used to make plots of %X59/Y34 of injuries garbage)
├── parameters.csv              # the holy grail - a cached file of the model parameters to feed to the grid search
```
## _misc_
Just like it sounds.
```
.
├── plot_rfs.py                 # post-grid search plots of changes in mean CCC given different random forest parameters
├── plot_xgb.py                 # post-grid search plots of changes in mean CCC given different GBT parameters
├── pull_garbage_packages.py    # archive of how I pulled the garbage packges shown in `maps/package_list.xlsx`
├── training_summaries.py       # creates vetting table - mean ccc across 500 test datasets for random forest and GBT
```

## _thesis_data_prep_
### Overview
Controlled through the `MCoDLauncher`, this folder houses the scripts to format all input data for the bag of words. Data is prepped (and saved) separately for X59/Y34, and you have the ability to choose which intermediate cause you want to launch (or both), along with flexibility in which sources to format (if not all of them). Current input data consists of all available multiple cause of death data in the GBD (with the exception of South Africa). The `MCauseLauncher` sources scripts that first standardize data in it's rawest form and formats it in a way complementary to multiple cause of death analysis done at IHME. Important output columns include all demographic-related information (age_group_id, sex_id, year_id, location_id), all ICD coded information (formatted as "cause" for the underlying cause of death, then "multiple_cause_x" for x causes in the chain) and some columns specific to IHME cause of death formatting and processing (nid, extract_type_id, code_system_id). Once this standardization is complete, formatting is done specifically for the bag of words model. Key steps are 1. Dropping rows where there is only a single multiple cause of death that is the same as the underlying cause of death 2. Dropping rows without causes in the chain 3. Removing any duplicated ICD codes for a given row 4. Subsetting to only injuries/X59/Y34 deaths 5. Removing any non N-codes in the chain. The `MCoDLauncher` also gives the flexibility to check the formatted datasets were saved in their expected folders.

```
.
├── launch_mcod_mapping.py      # houses the `MCoDLauncher` that launches all the formatting things
├── mcod_mapping.py             # houses an `MCoDMapper` class that standardizes the data for IHME multiple cause of death analyses
├── run_phase_format_map.py     # calls the `MCoDMapper` and performs all machine learning specific formatting
```

## _thesis_utils_
Some functions to help you along the way :)





-----
\* This project is dependent on multiple cause of death data belonging to the Institute for Health Metrics and Evaluation (IHME), sources IHME internal functions, and reads in files/sources a conda environment housed in IHME's internal file system.


