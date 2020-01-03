library(ISLR)
library(tree)
library(data.table)


# test from: https://www.datacamp.com/community/tutorials/decision-trees-R
carseats<-Carseats
High = ifelse(carseats$Sales<=8, "No", "Yes")
carseats = data.frame(carseats, High)

tree.carseats = tree(High~.-Sales, data=carseats)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats, pretty = 0)


# sepsis data
df <- fread("/ihme/cod/prep/mcod/process_data/sepsis_nonfatal_model/01_prep_regression/2018-11-05/all_locations_redistributed.csv")
df$cases_explicit[is.na(df$cases_explicit)] <- 0
df$cases_implicit[is.na(df$cases_implicit)] <- 0
df$deaths_explicit[is.na(df$deaths_explicit)] <- 0
df$deaths_implicit[is.na(df$deaths_implicit)] <- 0


df[, sepsis_cases := cases_implicit + cases_explicit]
df[, sepsis_deaths := deaths_implicit + deaths_explicit]
df[, sepsis_cfr := sepsis_deaths/sepsis_cases]

df[, c("sex_id", "age_group_id", "level_1",
         "level_2") := lapply(.SD, as.factor), .SDcols = c(
           "sex_id", "age_group_id", "level_1", "level_2")]
group_cols = c('sex_id', 'age_group_id', 'haqi', 'location_id', 'year_id', 'level_1', 'level_2')
df[, c("sex_id", "age_group_id") := lapply(.SD, as.factor), .SDcols = c("sex_id", "age_group_id")]


df = df[, lapply(.SD,sum), 
        by=group_cols, 
        .SDcols="sepsis_cfr"]
df[,haqi_mean_scaled:=haqi/100]

tree.sepsis = tree(sepsis_cfr~age_group_id+sex_id+haqi+level_2, df)
# tree.sepsis = tree(sepsis_cfr~age_group_id+sex_id+haqi_mean_scaled, df)
summary(tree.sepsis)
plot(tree.sepsis)
text(tree.sepsis, pretty = 0)

set.seed(101)
train=sample(1:nrow(df), 250)
tree.sepsis_train = tree(sepsis_cfr~age_group_id+sex_id+haqi, df=df, subset=train)
plot(tree.sepsis_train)
text(tree.sepsis_train, pretty = 0)


# doesnt work
# tree.pred = predict(tree.sepsis_train, df[-train,], type="tree")
# with(df[-train,], table(tree.pred, High))