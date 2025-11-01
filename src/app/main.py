# main.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths to data
train_path = "src/data/train.csv"
test_path = "src/data/test.csv"
submission_path = "src/data/gender_submission.csv"

# -----------------------
# Load datasets
# -----------------------
print("Loading training data...")
train_df = pd.read_csv(train_path)
print("Train dataset shape:", train_df.shape)
print(train_df.head())

# -----------------------
# Handle missing values
# -----------------------
print("\nHandling missing values in train dataset...")
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
print("Missing values handled")

# -----------------------
# Feature engineering
# -----------------------
print("\nEncoding categorical variables...")
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# HasCabin feature
train_df['HasCabin'] = train_df['Cabin'].notna().astype(int)
print("\nAdded HasCabin feature:")
print(train_df[['Cabin', 'HasCabin']].head())

# Extract title from Name
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3}
train_df['Title'] = train_df['Title'].map(title_mapping).fillna(4)
print("\nAdded Title feature:")
print(train_df[['Name', 'Title']].head())

# -----------------------
# Select features & target
# -----------------------
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'HasCabin', 'Title']
X = train_df[features]
y = train_df['Survived']

# -----------------------
# Train-validation split
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# Train logistic regression
# -----------------------
print("\nTraining logistic regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
print("Training accuracy:", train_acc)
print("Validation accuracy:", val_acc)

# -----------------------
# Load test dataset
# -----------------------
print("\nLoading test dataset...")
test_df = pd.read_csv(test_path)
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna('S')
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_df['HasCabin'] = test_df['Cabin'].notna().astype(int)
test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Title'].map(title_mapping).fillna(4)

X_test = test_df[features]

# -----------------------
# Predict on test set
# -----------------------
print("\nPredicting on test dataset...")
y_test_pred = model.predict(X_test)
print("First 10 predictions:", y_test_pred[:10])

print("\nPredicting on test dataset...")
y_test_pred = model.predict(X_test)

# Save predictions to a CSV file
output_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_test_pred
})
output_path = "src/data/test_predictions.csv"
output_df.to_csv(output_path, index=False)
print(f"Test predictions saved to {output_path}")


# -----------------------
# Print summary statistics
# -----------------------
print("\nPrediction summary:")
print(pd.Series(y_test_pred).value_counts())

# -----------------------
# Compare with sample submission
# -----------------------
print("\nLoading sample submission for comparison...")
submission_df = pd.read_csv(submission_path)
if 'Survived' in submission_df.columns:
    comparison = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Predicted': y_test_pred,
        'SampleSubmission': submission_df['Survived']
    })
    test_acc = accuracy_score(submission_df['Survived'], y_test_pred)
    print("\nFirst 10 comparison rows:")
    print(comparison.head(10))
    print("\nTest accuracy compared to sample submission:", test_acc)
else:
    print("Sample submission file does not contain 'Survived' column.")
    
