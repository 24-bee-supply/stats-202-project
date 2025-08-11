library(caret)
library(doParallel)

set.seed(202)

cl <- makeCluster(11)  
registerDoParallel(cl)

# Prepare data
X_train <- model.matrix(relevance ~ . - 1, data = train_transformed)
y_train <- as.factor(train_transformed$relevance)

# Specify larger k values to test
# Start higher since large k is doing better
k_values <- data.frame(
 k = seq(143, 159, by = 2)
)

# Run KNN with 10-fold CV
# Add preProcess to scale within each fold (prevents data leakage)
knn_model <- train(
 x = X_train,
 y = y_train,
 method = "knn",
 preProcess = c("center", "scale"),  # Scale within each CV fold
 trControl = trainControl(method = "cv", number = 10),
 tuneGrid = k_values
)

# Save model
saveRDS(knn_model, "../models/knn_model.rds")
stopCluster(cl)

# Print basic info
cat("Best k:", knn_model$bestTune$k, "\n")
cat("CV Accuracy:", max(knn_model$results$Accuracy), "\n")
