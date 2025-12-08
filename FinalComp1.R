library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)
library(ranger)
library(stacks)
library(glmnet)
library(doParallel)  # Added for parallel processing to speed up tuning

# Set up parallel workers (use most of your cores; adjust based on your machine)
cl <- makePSOCKcluster(detectCores() - 1)  # Leave 1 core free
registerDoParallel(cl)

train <- vroom::vroom("~/Downloads/FinalKaggleCompetition/DontGetKicked/training.csv",
                      na = c("", "NA", "NULL", "NOT AVAIL"))
test <- vroom::vroom("~/Downloads/FinalKaggleCompetition/DontGetKicked/test.csv",
                     na = c("", "NA", "NULL", "NOT AVAIL"))

train <- train %>% mutate(IsBadBuy = as.factor(IsBadBuy))

# Improved Recipe: Keep MMR columns and engineer price diffs/ratios for better predictions.
# Changed imputation to numerics only. Lowered other() threshold to group rare categories faster.
# Removed fewer columns (kept Model/SubModel/Trim with target encoding to capture info).
my_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  update_role(RefId, new_role = "ID") %>%
  update_role_requirements("ID", bake = FALSE) %>%
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_rm(contains("MMR")) %>%
  step_rm(BYRNO, WheelTypeID, VehYear, VNST, VNZIP1, PurchDate,
          AUCGUART, PRIMEUNIT,
          Model, SubModel, Trim) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_mean(all_nominal_predictors()) %>%
  step_zv()

prep <- prep(my_recipe)
bake <- bake(prep, new_data = train)

# Reduced folds to 3 for speed (still reliable for tuning)
folds <- vfold_cv(train, v = 3, repeats = 1)

# Random Forest: Smaller grid (levels=3), fewer trees for speed. Use random search for efficiency.
rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 200  # Reduced from 500
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>%
  add_model(rf_model)

# Switch to grid_random for faster/better exploration (size=10 instead of 5x5=25)
rf_tuning_grid <- grid_random(mtry(range = c(1, ncol(bake) - 1)),
                              min_n(),
                              size = 10)

# Tune with parallel (control_grid enables it)
untuned_control <- tune::control_grid(save_pred = TRUE, save_workflow = TRUE, parallel_over = "everything")

rf_models <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = rf_tuning_grid,
            metrics = metric_set(gain_capture),  # Proxy for Gini
            control = untuned_control,
            verbose = TRUE)

# LightGBM: Smaller grid (levels=3), add class weights for imbalance. Random search.
lgb_model <- boost_tree(
  tree_depth = tune(),
  trees = tune(),
  learn_rate = tune()
) %>%
  set_engine("lightgbm", 
             class_weights = "balanced"  # Handles imbalance for better Gini
  ) %>%
  set_mode("classification")

lgb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(lgb_model)

# Random grid, smaller size
lgb_tuning_grid <- grid_random(tree_depth(),
                               trees(range = c(50, 500)),  # Constrain trees for speed
                               learn_rate(),
                               size = 15)  # 15 points instead of 125

lgb_models <- lgb_wf %>%
  tune_grid(resamples = folds,
            grid = lgb_tuning_grid,
            metrics = metric_set(gain_capture),
            control = untuned_control,
            verbose = TRUE)

# Stacking: Fixed references (no ridge). Blend predictions.
my_stack <- stacks() %>%
  add_candidates(rf_models) %>%  # Fixed names
  add_candidates(lgb_models)

stack_mod <- my_stack %>%
  blend_predictions() %>%  # Blends weights
  fit_members()  # Fits on full train

# Predictions
stack_preds <- stack_mod %>%
  predict(new_data = test, type = "prob") %>%
  bind_cols(., test) %>% 
  select(RefId, .pred_1) %>% 
  rename(IsBadBuy = .pred_1)

vroom_write(x = stack_preds, file = "./submission_stack.csv", delim = ",")

# Clean up parallel
stopCluster(cl)