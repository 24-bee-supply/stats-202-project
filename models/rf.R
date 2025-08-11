library(ranger)
library(caret)

set.seed(202)

cat("=== Starting Random Forest Training ===\n")
cat("Time:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# Prepare data
X_train <- train_transformed %>% select(-relevance)
y_train <- as.factor(train_transformed$relevance)
cat("Data prepared:", nrow(X_train), "rows,", ncol(X_train), "features\n\n")

# Grid search with caret
cat("--- Grid Search with Cross-Validation ---\n")

# Set up grid
default_mtry <- floor(sqrt(ncol(X_train)))
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, default_mtry),
  splitrule = "gini",
  min.node.size = c(1, 5, 10)
)

cat("Grid search parameters:\n")
cat("- mtry values:", unique(rf_grid$mtry), "\n")
cat("- min.node.size values:", unique(rf_grid$min.node.size), "\n")
cat("- Total combinations:", nrow(rf_grid), "\n")
cat("- CV folds: 10\n")
cat("- Total models to fit:", nrow(rf_grid) * 10, "\n\n")

cat("Starting grid search...\n")
start_time <- Sys.time()

# Train with CV
rf_model <- train(
  x = X_train,
  y = y_train,
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 10,
    allowParallel = FALSE,
    verboseIter = TRUE
  ),
  tuneGrid = rf_grid,
  num.trees = 500,
  importance = "impurity",
  num.threads = parallel::detectCores() - 1,
  verbose = FALSE
)

end_time <- Sys.time()
cat("\nGrid search completed in", round(difftime(end_time, start_time, units = "mins"), 1), "minutes\n\n")

# Save model
cat("Saving model...\n")
saveRDS(rf_model, "../models/rf_ranger_model.rds")
cat("Model saved to ../models/rf_ranger_model.rds\n\n")

# Print results
cat("=== RESULTS ===\n")
cat("Best mtry:", rf_model$bestTune$mtry, "\n")
cat("Best min.node.size:", rf_model$bestTune$min.node.size, "\n")
cat("Best splitrule:", rf_model$bestTune$splitrule, "\n")
cat("CV Accuracy:", round(max(rf_model$results$Accuracy), 4), "\n\n")

# Print all results sorted by accuracy
cat("--- All Grid Search Results ---\n")
print(rf_model$results[order(-rf_model$results$Accuracy), 
                       c("mtry", "min.node.size", "Accuracy", "AccuracySD")])

cat("\n=== Training Complete ===\n")