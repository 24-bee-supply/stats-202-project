library(xgboost)
library(tidyverse)

# Load test data
test <- read.csv("../data/test.csv")

# Apply same preprocessing and transformations
test_processed <- preprocess_data(test)
test_transformed <- apply_transformations(test_processed)

# Load the XGBoost model
xgb_results <- readRDS("../models/xgb_model.rds")
xgb_model <- xgb_results$model

# Prepare test data for XGBoost (needs matrix format)
test_matrix <- model.matrix(~ . - 1, data = test_transformed)

# Create DMatrix for prediction
dtest <- xgb.DMatrix(data = test_matrix)

# Make predictions (returns probabilities)
predictions_prob <- predict(xgb_model, dtest)

# Convert to binary (threshold at 0.5)
predictions_binary <- as.integer(predictions_prob > 0.5)

# Check prediction distribution
cat("XGBoost Prediction Distribution:\n")
cat("===================================\n")
cat("Proportion of 1s:", round(mean(predictions_binary), 3), "\n")
cat("Proportion of 0s:", round(1 - mean(predictions_binary), 3), "\n\n")

# Create submission
submission <- data.frame(
  id = test$id,
  relevance = predictions_binary
)

# Save predictions
write.csv(submission, "../data/bue-alex-xgboost-predictions.csv", row.names = FALSE)

cat("Predictions saved to: bue-alex-xgboost-predictions.csv\n\n")

# Show first 10 predictions with probabilities
cat("First 10 predictions:\n")
submission_preview <- data.frame(
  id = test$id[1:10],
  probability = round(predictions_prob[1:10], 3),
  prediction = predictions_binary[1:10]
)
print(submission_preview)