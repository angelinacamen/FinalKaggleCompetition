library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(lme4)
library(parsnip)
library(discrim)
library(kernlab)
library(themis)
library(kernlab)
library(dbarts)

# Set up parallel workers (use most of your cores; adjust based on your machine)
cl <- makePSOCKcluster(detectCores() - 1)  # Leave 1 core free
registerDoParallel(cl)

train <- vroom::vroom("~/Downloads/FinalKaggleCompetition/DontGetKicked/training.csv",
                      na = c("", "NA", "NULL", "NOT AVAIL"))
test <- vroom::vroom("~/Downloads/FinalKaggleCompetition/DontGetKicked/test.csv",
                     na = c("", "NA", "NULL", "NOT AVAIL"))

train <- train %>% mutate(IsBadBuy = as.factor(IsBadBuy))

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

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)


bart_model <- parsnip::bart(trees = 375) %>% 
  set_engine("dbarts") %>% # need to install
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)


bart_tuneGrid <- grid_regular(trees(),
                              levels = 6)

## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats = 1)


tuned_bart <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = bart_tuneGrid,
            metrics = metric_set(accuracy))
## Predict

final_wf <-
  bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

numeric_cols <- c(
  "MMRCurrentAuctionAveragePrice",
  "MMRCurrentAuctionCleanPrice",
  "MMRCurrentRetailAveragePrice",
  "MMRCurrentRetailCleanPrice"
)

test[numeric_cols] <- lapply(test[numeric_cols], as.numeric)


kicked_bart_predictions <- 
  predict(final_wf,
          new_data = test,
          type = "prob") %>% 
  bind_cols(test) %>% 
  rename(IsBadBuy = .pred_1) %>%
  select(RefId, IsBadBuy)

vroom_write(x = kicked_bart_predictions, 
            file = "./submission.csv", delim = ",")
