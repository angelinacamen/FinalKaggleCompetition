## Load Packages
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(lme4)
library(parsnip)
library(discrim)
library(kernlab)
library(themis)
library(xgboost)
library(doParallel)

# Set up parallel workers
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

# Load the data
train <- vroom::vroom("~/Downloads/FinalKaggleCompetition/DontGetKicked/training.csv",
                      na = c("", "NA", "NULL", "NOT AVAIL"))
test <- vroom::vroom("~/Downloads/FinalKaggleCompetition/DontGetKicked/test.csv",
                     na = c("", "NA", "NULL", "NOT AVAIL"))

train <- train %>% mutate(IsBadBuy = as.factor(IsBadBuy))

## Create Recipe (same as yours)
my_recipe <- recipe(IsBadBuy ~ ., data = train) %>% 
  update_role(RefId, new_role = 'ID') %>% 
  update_role_requirements('ID', bake = FALSE) %>% 
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) %>% 
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>% 
  step_mutate_at(all_nominal_predictors(), fn = factor) %>% 
  step_rm(BYRNO, WheelTypeID, VehYear, VNST, PurchDate, AUCGUART, PRIMEUNIT,
          Model, SubModel, Trim) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.7) %>% 
  step_other(all_nominal_predictors(), threshold = 0.09) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_impute_median(all_numeric_predictors())

## XGBoost Model 
xgb_model <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  tree_depth = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

## Workflow
xgb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(xgb_model)

## Cross-Validation
folds <- vfold_cv(train, v = 5)

## Tuning Grid 
xgb_grid <- grid_latin_hypercube(
  trees(range = c(400, 1200)),
  learn_rate(range = c(0.01, 0.15)),
  mtry(range = c(5, 25)),
  tree_depth(range = c(3, 8)),
  size = 15
)

## Tune Model
tuned_xgb <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid = xgb_grid,
  metrics = metric_set(accuracy)
)

best_xgb <- select_best(
  tuned_xgb,
  metric = "accuracy"
)

## Finalize Workflow
final_xgb_wf <- finalize_workflow(xgb_wf, best_xgb)

## Fit Final Model on ALL Training Data
final_xgb_fit <- fit(final_xgb_wf, data = train)

## Ensure numeric columns in test set
numeric_cols <- c(
  "MMRCurrentAuctionAveragePrice",
  "MMRCurrentAuctionCleanPrice",
  "MMRCurrentRetailAveragePrice",
  "MMRCurrentRetailCleanPrice"
)
test[numeric_cols] <- lapply(test[numeric_cols], as.numeric)

## Generate Predictions
xgb_predictions <- predict(final_xgb_fit, new_data = test, type = "prob") %>%
  bind_cols(test) %>%
  rename(IsBadBuy = .pred_1) %>%
  select(RefId, IsBadBuy)

## Write Kaggle Submission File
vroom_write(
  x = xgb_predictions,
  file = "./submission_xgb.csv",
  delim = ","
)

stopCluster(cl)
