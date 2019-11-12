# 11/11/2019
# Just getting a feel for rtree vs rpart modeling (and predicting)
# Note: There's no cross-validation/train-test datasets here so we're real 
# rudimentary today girl
# wait rpart does cross-validation for you...?

rm(list=ls())

library(data.table)
library(tree, lib.loc="/homes/agesak/R/3.5")
library(rpart, lib.loc="/homes/agesak/R/3.5")
library(rpart.plot, lib.loc="/homes/agesak/R/3.5")

functions_dir <- "/ihme/cc_resources/libraries/current/r/"

#source central functions
source(paste0(functions_dir, "get_demographics.R"))
source(paste0(functions_dir, "get_covariate_estimates.R"))
source(paste0(functions_dir, "get_location_metadata.R"))

# use the central functions
dems <- get_demographics(gbd_team="cod",gbd_round_id=5)
haqi_val <- get_covariate_estimates(covariate_id = 1099, gbd_round_id = 5)
haqi_val <- haqi_val[, c('location_id', 'year_id', 'mean_value')]
setnames(haqi_val, "mean_value", "haqi")

# FUNCTIONS -------------------------------------------------------------------------------
create_template <- function(dems, year) {
  ##create template for prediction
  locs = dems$location_id
  sexes = dems$sex_id
  ages = dems$age_group_id
  #ages = ages[-1:-3] #dropping neonates
  year = as.numeric(year)
  template = expand.grid(year,locs,ages,sexes)
  setnames(template, c('Var1', 'Var2', 'Var3', 'Var4'), c('year_id', 'location_id', 'age_group_id', 'sex_id'))
  template = as.data.table(template)
  return(template)
}

# DO THE STUFF ---------------------------------------------------------------------------------------

# read in dataset used in GBD 2019 final model
df <- fread("/ihme/cod/prep/mcod/process_data/x59/2019_03_07/model_input.csv")

# tree package --------------------------------------------------------------
tree.inj = tree(x59_fraction~age_group_id+sex_id+haqi+cause_id, df)
plot(tree.inj)
text(tree.inj, pretty = 0)

# okay now rpart ------------------------------------------------------------
rpart_tree <- rpart(
  x59_fraction~age_group_id+sex_id+haqi+cause_id,
  df,
  method = "anova")
summary(rpart_tree)
# so pretty 
rpart.plot(rpart_tree)
# plot cross-validation
plotcp(rpart_tree)
# 0.694 press statistic - https://en.wikipedia.org/wiki/PRESS_statistic
rpart_tree$cptable

temp <- create_template(dems, 2016)
temp <- merge(temp, haqi_val, by=c('location_id', 'year_id'))
temp[, cause_id := 690]
# predicts one value for everyone...
temp[, predict := predict(rpart_tree, newdata = temp)]
unique(temp$predict)
