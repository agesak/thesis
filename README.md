# Machine Learning for Injuries Cause of Death Assignment: A New Method for the Global Burden of Disease Study
### _A master's thesis completed for the University of Washington Master of Public Health Program (2020)_
## Kareha Agesa

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
To activate virtual environment...
```bash

```


## **Folder: analysis**
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
- launch_training_model: fits a given classifier on the training data (and performs 5 fold cross validation using)
- launch_testing_models: predict on each of the 500 generated test datasets for a given classifier
- launch_int_cause_predictions: refit on all observed data and predict on the unobserved data

\* The order of these phases is not defined in the code, but you should start by creating the train/test datasets, then if you like you could either launch the training models or create the test datasets. Launching the testing models must come after the 500 generated test datasets were created in phase: create_test_datasets, then you can launch_int_cause_predictions.

### _Machine Learning Implementations_
1. Bag of words - [CountVectorizer in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
2. [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) 
2. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. [Gradient Boosted Trees](https://xgboost.readthedocs.io/en/latest/python/index.html)
4. Deep Neural Network (https://keras.io/)

### Various Data/Model Attributes:
The `ModelLauncher` class offers various flexibility regarding the input data and model attributes. For the input data, examples include the ability to model at either the most detailed (or country) location level and the ability to subset to only data for a given ICD system (either ICD 9 or ICD 10). The class also offers the ability to select various attributes of the data as features in the bag of words. For example, keeping standard the inclusion of ICD codes as features, models can additionally be run with just or age, sex, location, and year as features. The ability also exists to run separate models by age. As a supplementary analysis, the hierarchial nature of the ICD was explored as features in the bag of words. For example, instead of just including the most detailed ICD code given in the data, models were tried with just 3 digit ICD codes, 3 digit ICD codes and the ICD letter (for ICD 10), and most detailed code and the letter (for ICD 10). These are denoted as "most_detailed" "aggregate_only", "aggregate_and_letter", and "most_detailed_and_letter".


## manuscript
### Overview
```
.
├── appendix_plot_rd_props.R
├── by_country.R
├── compare_rd_results.py
├── country_cause_table.py
├── get_best_model_params.py
├── number_plugging.py
├── percent_garbage.py
├── plot_rd_props.R
```

## maps
### Overview

```
.
├── injuries_overrrides.csv
├── package_list.xlsx
├── parameters.csv
```
## misc
Just like it sounds
```
.
├── plot_rfs.py
├── plot_xgb.py
├── pull_garbage_packages.py
├── training_summaries.py
```

## thesis_data_prep
### Overview

```
.
├── launch_mcod_mapping.py
├── mcod_mapping.py
├── run_phase_format_map.py
```

## thesis_utils
### Overview

```
.
├── clf_switching.py
├── directories.py
├── grid_search.py
├── misc.py
├── model_evaluation.py
├── modeling.py
```
-----
\* This project is dependent on multiple cause of death data belonging to IHME, sources IHME internal functions, and reads in files housed in IHME's internal file system.


