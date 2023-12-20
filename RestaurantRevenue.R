library(tidyverse)
library(tidymodels)
library(vroom)
library(ranger)

train <- vroom("./RestaurantRevenue/train.csv")
test <- vroom("./RestaurantRevenue/test.csv")

train$date <- as.Date(strptime(train$`Open Date`, "%m/%d/%Y"))
test$date <- as.Date(strptime(test$`Open Date`, "%m/%d/%Y"))

my_recipe <- recipe(revenue~., data = train) %>%
  step_mutate_at(`City Group`, fn = factor) %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month") %>%
  step_date(date, features="year") %>%
  step_rm(`Open Date`, Id, City, Type, date)

prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = train)

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rand_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range = c(1,(ncol(baked)-1))),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- rand_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq))

bestTune <- CV_results %>%
  select_best("rmse")

final_wf <- rand_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

preds <- final_wf %>%
  predict(new_data = test)

final_preds <- cbind(test$Id, preds)
colnames(final_preds) <- c("Id","Prediction")

vroom_write(final_preds, "./RestaurantRevenue/preds.csv", ",")


