library(ranger)
library(caret)

set.seed(202)

# Prepare data
X_train <- train_transformed %>% select(-relevance)
y_train <- as.factor(train_transformed$relevance)

# OPTION 1: Quick single ranger model (no CV, takes ~30 seconds)
# This is often good enough and much faster
quick_rf <- ranger(
  relevance ~ ., 
  data = train_transformed,
  num.trees = 500,
  mtry = floor(sqrt(ncol(X_train))),  
  min.node.size = 5,
  importance = "impurity",
  probability = TRUE,  # Get probabilities for threshold tuning
  num.threads = parallel::detectCores() - 1
)

cat("Quick RF OOB Error:", quick_rf$prediction.error, "\n")

# OPTION 2: Minimal grid search with caret (5-10 minutes)
# Smaller, smarter grid
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, mtry = floor(sqrt(ncol(X_train)))),  # Just 3 values
  splitrule = "gini",  # Just use gini (usually best)
  min.node.size = c(1, 5, 10)  # Just 2 values
)

# Use 5-fold CV instead of 10 for speed
rf_model <- train(
  x = X_train,
  y = y_train,
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 10,  # 10-fold CV
    allowParallel = FALSE  # Let ranger handle parallelization internally
  ),
  tuneGrid = rf_grid,
  num.trees = 500,  # 500 trees as requested
  importance = "impurity",
  num.threads = parallel::detectCores() - 1,  # Use all cores within ranger
  verbose = FALSE
)

# Save both models
saveRDS(list(quick_model = quick_rf, 
             cv_model = rf_model), 
        "../models/rf_ranger_model.rds")

# Print results
cat("\n--- Quick Model (OOB) ---\n")
cat("OOB Accuracy:", 1 - quick_rf$prediction.error, "\n")

cat("\n--- CV Model ---\n")
cat("Best mtry:", rf_model$bestTune$mtry, "\n")
cat("Best min.node.size:", rf_model$bestTune$min.node.size, "\n")
cat("CV Accuracy:", max(rf_model$results$Accuracy), "\n")