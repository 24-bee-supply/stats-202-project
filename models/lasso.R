library(glmnet)

set.seed(202)

# Prepare data
X_train <- model.matrix(relevance ~ . - 1, data = train_transformed)

y_train <- train_transformed$relevance

# Run LASSO with 10-fold CV
cv_lasso <- cv.glmnet(
  x = X_train,
  y = y_train,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  type.measure = "class"
)

# Fit final model with optimal lambda
lasso_final <- glmnet(X_train, y_train, family = "binomial", 
                      alpha = 1, lambda = cv_lasso$lambda.min)

# Save results
lasso_results <- list(cv_model = cv_lasso, final_model = lasso_final)
saveRDS(lasso_results, "lasso_model.rds")

# Print basic info
cat("Best lambda:", cv_lasso$lambda.min, "\n")
cat("CV Accuracy:", 1 - min(cv_lasso$cvm), "\n")