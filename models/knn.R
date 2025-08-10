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
 k = c(41, 51, 71, 101, 151, 201, 251, 301, 401, 451)
)

# Run KNN with 10-fold CV
knn_model <- train(
  x = X_train,
  y = y_train,
  method = "knn",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = k_values  # Use specific k values instead of tuneLength
)

# Save model
saveRDS(knn_model, "knn_model.rds")
stopCluster(cl)