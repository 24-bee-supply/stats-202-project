library(xgboost)

set.seed(202)

# Prepare data
X_train <- train_transformed %>% select(-relevance)
y_train <- train_transformed$relevance
X_train_matrix <- model.matrix(~ . - 1, data = X_train)

# Create DMatrix
dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train)

# Cross-validation to find best nrounds
params <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 150,
  nfold = 10,
  early_stopping_rounds = 10,
  verbose = FALSE
)

# Train final model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = cv$best_iteration,
  verbose = 0
)

# Save
saveRDS(list(model = xgb_model, cv = cv), "../models/xgb_model.rds")

# Print result
cat("XGBoost CV Accuracy:", round(1 - cv$evaluation_log$test_error_mean[cv$best_iteration], 4), "\n")
cat("Best nrounds:", cv$best_iteration, "\n")