# Titanic Survival Prediction Model
# Author: [Your Name]
# Description: Logistic regression model to predict passenger survival

# =============================================================================
# SETUP
# =============================================================================

# Load libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(caret)
  library(stringr)
})

# Configuration
CONFIG <- list(
  train_path = "src/data/train.csv",
  test_path = "src/data/test.csv",
  submission_path = "src/data/gender_submission.csv",
  seed = 42,
  train_split = 0.8,
  threshold = 0.5
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Handle missing values in dataset
#' @param df Dataframe to process
#' @return Dataframe with imputed values
handle_missing_values <- function(df) {
  df$Age[is.na(df$Age)] <- median(df$Age, na.rm = TRUE)
  df$Fare[is.na(df$Fare)] <- median(df$Fare, na.rm = TRUE)
  df$Embarked[is.na(df$Embarked)] <- "S"
  return(df)
}

#' Extract title from passenger name
#' @param name_vector Vector of passenger names
#' @return Numeric vector of encoded titles
extract_title <- function(name_vector) {
  titles <- str_extract(name_vector, " ([A-Za-z]+)\\.")
  title_mapping <- c("Mr" = 0, "Miss" = 1, "Mrs" = 2, "Master" = 3)
  
  encoded_titles <- sapply(titles, function(title) {
    if (title %in% names(title_mapping)) {
      return(title_mapping[title])
    } else {
      return(4)  # Other titles
    }
  })
  
  return(unname(encoded_titles))
}

#' Engineer features for the dataset
#' @param df Dataframe to process
#' @return Dataframe with engineered features
engineer_features <- function(df) {
  # Encode Sex: male=0, female=1
  df$Sex <- ifelse(df$Sex == "male", 0, 1)
  
  # Encode Embarked: S=0, C=1, Q=2
  df$Embarked <- recode(df$Embarked, "S" = 0, "C" = 1, "Q" = 2)
  
  # Binary indicator for cabin information
  df$HasCabin <- ifelse(!is.na(df$Cabin), 1, 0)
  
  # Extract and encode title from name
  df$Title <- extract_title(df$Name)
  
  return(df)
}

#' Calculate and display model accuracy
#' @param predictions Vector of predictions
#' @param actuals Vector of actual values
#' @param dataset_name Name of dataset for display
#' @return Accuracy score
calculate_accuracy <- function(predictions, actuals, dataset_name = "") {
  accuracy <- mean(predictions == actuals)
  cat(sprintf("%s Accuracy: %.4f (%.2f%%)\n", 
              dataset_name, accuracy, accuracy * 100))
  return(accuracy)
}

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

cat("\n========================================\n")
cat("LOADING AND PREPROCESSING DATA\n")
cat("========================================\n\n")

# Load training data
cat("Loading training data...\n")
train_df <- read_csv(CONFIG$train_path, show_col_types = FALSE)
cat(sprintf("Train dataset shape: %d rows, %d columns\n", 
            nrow(train_df), ncol(train_df)))

# Handle missing values
cat("\nHandling missing values...\n")
train_df <- handle_missing_values(train_df)

# Feature engineering
cat("Engineering features...\n")
train_df <- engineer_features(train_df)

# =============================================================================
# MODEL TRAINING
# =============================================================================

cat("\n========================================\n")
cat("MODEL TRAINING\n")
cat("========================================\n\n")

# Define features and target
FEATURES <- c("Pclass", "Sex", "Age", "SibSp", "Parch", 
              "Fare", "Embarked", "HasCabin", "Title")
X <- train_df[, FEATURES]
y <- train_df$Survived

# Train-validation split
set.seed(CONFIG$seed)
train_index <- createDataPartition(y, p = CONFIG$train_split, list = FALSE)
X_train <- X[train_index, ]
X_val <- X[-train_index, ]
y_train <- y[train_index]
y_val <- y[-train_index]

cat(sprintf("Training set: %d samples\n", length(y_train)))
cat(sprintf("Validation set: %d samples\n\n", length(y_val)))

# Train logistic regression model
cat("Training logistic regression model...\n")
model <- glm(y_train ~ ., data = X_train, family = binomial)

# Generate predictions
train_pred <- ifelse(predict(model, X_train, type = "response") > CONFIG$threshold, 1, 0)
val_pred <- ifelse(predict(model, X_val, type = "response") > CONFIG$threshold, 1, 0)

# Evaluate performance
cat("\n")
calculate_accuracy(train_pred, y_train, "Training")
calculate_accuracy(val_pred, y_val, "Validation")

# =============================================================================
# TEST SET PREDICTION
# =============================================================================

cat("\n========================================\n")
cat("TEST SET PREDICTION\n")
cat("========================================\n\n")

# Load and preprocess test data
cat("Loading test dataset...\n")
test_df <- read_csv(CONFIG$test_path, show_col_types = FALSE)
test_df <- handle_missing_values(test_df)
test_df <- engineer_features(test_df)

X_test <- test_df[, FEATURES]

# Generate predictions
cat("Generating predictions...\n")
y_test_pred <- ifelse(predict(model, X_test, type = "response") > CONFIG$threshold, 1, 0)

# Display prediction summary
cat("\nPrediction Summary:\n")
pred_table <- table(y_test_pred)
print(pred_table)
cat(sprintf("Survival rate: %.2f%%\n", 
            pred_table["1"] / sum(pred_table) * 100))

# Save predictions to CSV
output_df <- data.frame(
  PassengerId = test_df$PassengerId,
  Survived = y_test_pred
)
output_path <- "src/data/test_predictions_r.csv"
write.csv(output_df, output_path, row.names = FALSE)
cat("Test predictions saved to", output_path, "\n")

# =============================================================================
# COMPARISON WITH SAMPLE SUBMISSION
# =============================================================================

cat("\n========================================\n")
cat("SAMPLE SUBMISSION COMPARISON\n")
cat("========================================\n\n")

submission_df <- read_csv(CONFIG$submission_path, show_col_types = FALSE)

if ("Survived" %in% colnames(submission_df)) {
  comparison <- data.frame(
    PassengerId = test_df$PassengerId,
    Predicted = y_test_pred,
    Sample = submission_df$Survived,
    Match = y_test_pred == submission_df$Survived
  )
  
  cat("First 10 predictions:\n")
  print(head(comparison, 10))
  
  cat("\n")
  calculate_accuracy(y_test_pred, submission_df$Survived, "Test")
  
  # Show mismatches
  mismatches <- sum(!comparison$Match)
  cat(sprintf("\nMismatches: %d out of %d (%.2f%%)\n", 
              mismatches, nrow(comparison), 
              mismatches / nrow(comparison) * 100))
} else {
  cat("Sample submission file does not contain 'Survived' column.\n")
}

cat("\n========================================\n")
cat("ANALYSIS COMPLETE\n")
cat("========================================\n")