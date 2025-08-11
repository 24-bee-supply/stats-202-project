library(glmnet)

set.seed(202)

# Prepare data
X_train <- model.matrix(relevance ~ . - 1, data = train_transformed)
y_train <- train_transformed$relevance

# Run LASSO with 10-fold CV
# standardize = TRUE (default) ensures scaling happens within each CV fold
cv_lasso <- cv.glmnet(
  x = X_train,
  y = y_train,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  type.measure = "class",
  standardize = TRUE  # Explicitly set to TRUE (scales within each fold)
)

# Fit final model with optimal lambda
# Again with standardize = TRUE for consistency
lasso_final <- glmnet(X_train, y_train, 
                      family = "binomial", 
                      alpha = 1, 
                      lambda = cv_lasso$lambda.min,
                      standardize = TRUE)

# Save results
lasso_results <- list(cv_model = cv_lasso, final_model = lasso_final)
saveRDS(lasso_results, "../models/lasso_model.rds")

# Print basic info
cat("Best lambda:", cv_lasso$lambda.min, "\n")
cat("CV Accuracy:", 1 - min(cv_lasso$cvm), "\n")