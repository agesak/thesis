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


## analysis
### Overview
The analysis pipeline (creation of the train/test datasets, model fitting, and model evaluation) are launched using the `ModelLauncher` class in `launch_models.py`. This class accepts command line arguments related to which classifier to run, which intermediate cause (either X59 or Y34) to use, which phase of the pipeline (defined above), and various characteristics of the data/model (ie. should the data be at the most-detailed location level (as opposed to country)?, which attributes of the data would you like to include as features in the bag of words? (just age? or age, sex, location, and year?), do you want to run separate models by age? would you like to experiment with the hierachial nature of the ICD in your models, by also including the ICD chapter (for ICD 10), and less detailed 3 digit codes, in addition to most detailed codes). This folder is structured as follows:

```
.
├── create_test_datasets.py         # worker script to generate each of 500 test datasets from a Dirichlet distribution
├── launch_models.py                # main class that creates train/test datasets, launches each classifier, and performs model evaluation
├── run_models.py                   # fits each classifier on the training data
├── run_testing_predictions.py      # performs predictions on each of the 500 generated test datasets using each classifier
├── run_unobserved_predictions.py   # creates file of summary statistics across all evaluation metrics for the 500 generated test datasets, refits a given classifier on all unobserved data (train and test), predicts on unobserved data

```
### Model Phases
Each phase of this pipeline is defined as follows:
- train_test: creation of the (75%) train and (25%) test datasets
- create_test_datasets: creation of the 500 generated test datasets used for model evaluation
- launch_training_model: fits a given classifier on the training data (and performs 5 fold cross validation using)
- launch_testing_models: predict on each of the 500 generated test datasets for a given classifier
- launch_int_cause_predictions: refit on all observed data and predict on the unobserved data

### Bag of words:


### Classifiers:
1. 
2. 
3. 
4. 
5. 

### Model Evaluation:


### Various Data/Model Attributes:





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


