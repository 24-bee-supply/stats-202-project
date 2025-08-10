library(xgboost)
library(caret)

set.seed(202)

# Prepare data
X_train <- as.matrix(train_transformed %>% select(-relevance))
y_train <- train_transformed$relevance

# Run XGBoost with CV
boost_model <- train(
  x = X_train,
  y = y_train,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 3,  # Tests 3 parameter combinations
  objective = "binary:logistic",
  verbosity = 0
)

# Save model
saveRDS(boost_model, "boost_model.rds")

# Print basic info
print(boost_model)
cat("\nBest parameters:\n")
print(boost_model$bestTune)
cat("CV Accuracy:", max(boost_model$results$Accuracy), "\n")