library(caret)
library(glmnet)
library(ranger)

set.seed(202)

cat("=== Building Ensemble Model ===\n\n")

# Load saved models
lasso_results <- readRDS("../models/lasso_model.rds")
knn_model <- readRDS("../models/knn_model.rds")
rf_model <- readRDS("../models/rf_ranger_model.rds")

# Get training predictions
X_train <- model.matrix(relevance ~ .^2 - 1, data = train_transformed)  # WITH INTERACTIONS
y_train <- train_transformed$relevance

# 1. LASSO probabilities
lasso_probs <- predict(lasso_results$final_model, newx = X_train, type = "response")[,1]

# 2. KNN binary predictions
knn_binary <- as.integer(as.character(predict(knn_model, newdata = X_train)))

# 3. RF probabilities (train once for ensemble)
rf_final <- ranger(
  relevance ~ .,
  data = train_transformed,
  num.trees = 500,
  mtry = rf_model$bestTune$mtry,
  min.node.size = rf_model$bestTune$min.node.size,
  probability = TRUE,
  num.threads = parallel::detectCores() - 1
)
rf_probs <- rf_final$predictions[, 2]

# Best ensemble: weighted average
ensemble_probs <- 0.4 * lasso_probs + 0.5 * rf_probs + 0.1 * knn_binary

# Find optimal threshold
thresholds <- seq(0.45, 0.55, by = 0.01)
accuracies <- sapply(thresholds, function(t) mean((ensemble_probs >= t) == y_train))
optimal_threshold <- thresholds[which.max(accuracies)]

cat("Individual Model Accuracies:\n")
cat("LASSO:", round(mean((lasso_probs >= 0.5) == y_train), 4), "\n")
cat("RF:", round(mean((rf_probs >= 0.5) == y_train), 4), "\n")
cat("KNN:", round(mean(knn_binary == y_train), 4), "\n")
cat("\nEnsemble Accuracy:", round(max(accuracies), 4), "\n")
cat("Optimal threshold:", optimal_threshold, "\n")

# Save ensemble
ensemble_model <- list(
  lasso_model = lasso_results$final_model,
  rf_final = rf_final,
  knn_model = knn_model,
  threshold = optimal_threshold
)
saveRDS(ensemble_model, "../models/ensemble_model.rds")

cat("\n=== Making Test Predictions ===\n")

# Load and prepare test data
test <- read.csv("../data/test.csv")
test_processed <- preprocess_data(test)
test_transformed <- apply_transformations(test_processed)

# Get test predictions
X_test <- model.matrix(~ .^2 - 1, data = test_transformed)  # WITH INTERACTIONS (same as training)
test_lasso <- predict(ensemble_model$lasso_model, newx = X_test, type = "response")[,1]
test_rf <- predict(ensemble_model$rf_final, data = test_transformed)$predictions[, 2]
test_knn <- as.integer(as.character(predict(ensemble_model$knn_model, newdata = X_test)))

# Ensemble prediction
test_ensemble <- 0.4 * test_lasso + 0.5 * test_rf + 0.1 * test_knn
predictions <- as.integer(test_ensemble >= ensemble_model$threshold)

# Save submission
submission <- data.frame(
  id = test$id,
  relevance = predictions
)
write.csv(submission, "../data/ensemble_submission.csv", row.names = FALSE)

cat("Predictions saved. Distribution: ", mean(predictions), "positive\n")