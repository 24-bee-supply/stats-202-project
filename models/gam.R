
library(mgcv)
library(caret)

# Set seed for reproducibility
set.seed(202)

# Use the transformed data
gam_data <- train_transformed

# Build GAM formula with smooth terms for continuous variables
# Using the actual variable names in train_transformed
# For log-transformed variables, we can still apply smoothing
# Binary/dummy variables are included as parametric terms
gam_formula <- relevance ~ 
  s(sig1, k = 10) + 
  s(sig2, k = 10) + 
  s(log_sig3, k = 10) + 
  s(log_sig4, k = 10) + 
  s(log_sig5, k = 10) + 
  s(log_sig6_positive, k = 10) + 
  s(sig7, k = 10) + 
  s(sig8, k = 10) + 
  sig6_is_zero +
  is_homepage_1 + 
  query_length_2 +
  query_length_3 +
  query_length_4 +
  query_length_5 +
  query_length_6 +
  query_length_7 +
  query_length_8

# Fit GAM model
gam_model <- gam(
  gam_formula,
  data = gam_data,
  family = binomial(link = "logit"),
  method = "REML"  # Recommended for smoother selection
)

# Model summary
cat("GAM Model Summary\n")
cat("==================\n")
summary_gam <- summary(gam_model)
cat("Deviance explained:", round(summary_gam$dev.expl * 100, 2), "%\n")
cat("AIC:", AIC(gam_model), "\n\n")

# Check which smooth terms are significant
cat("Significance of smooth terms:\n")
print(summary_gam$s.table)
cat("\nParametric coefficients:\n")
print(summary_gam$p.table)

# 10-fold cross-validation for GAM
folds <- createFolds(gam_data$relevance, k = 10, list = TRUE)
cv_accuracy <- numeric(10)

cat("\n10-Fold Cross-Validation:\n")
cat("==========================\n")

for(i in 1:10) {
  # Split data
  train_idx <- unlist(folds[-i])
  test_idx <- folds[[i]]
  
  train_fold <- gam_data[train_idx, ]
  test_fold <- gam_data[test_idx, ]
  
  # Fit model on training fold
  fold_model <- gam(
    gam_formula,
    data = train_fold,
    family = binomial(link = "logit"),
    method = "REML"
  )
  
  # Predict on test fold
  pred_probs <- predict(fold_model, newdata = test_fold, type = "response")
  pred_class <- ifelse(pred_probs > 0.5, 1, 0)
  
  # Calculate accuracy
  accuracy <- mean(pred_class == test_fold$relevance)
  cv_accuracy[i] <- accuracy
  
  cat("Fold", sprintf("%2d", i), ": Accuracy =", round(accuracy, 4), "\n")
}

cat("\nCross-Validation Summary:\n")
cat("Mean CV Accuracy:", round(mean(cv_accuracy), 4), "\n")
cat("SD of CV Accuracy:", round(sd(cv_accuracy), 4), "\n")

# Plot smooth terms to visualize non-linear relationships
par(mfrow = c(3, 3))
plot(gam_model, pages = 1, residuals = TRUE, rug = TRUE, 
     cex.lab = 1.2, shade = TRUE, shade.col = "lightblue",
     main = "GAM Smooth Terms")
par(mfrow = c(1, 1))

# Check for concurvity (multicollinearity for GAMs)
cat("\nConcurvity Check:\n")
concurv <- concurvity(gam_model, full = TRUE)
print(round(concurv$worst, 3))