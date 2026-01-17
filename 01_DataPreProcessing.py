# TASK 1: DATA PREPROCESSING FOR MACHINE LEARNING
# Dataset: Churn Prediction Data

# STEP 1: Import Required Libraries
# pandas -> data handling
# numpy -> numerical operations
# sklearn -> preprocessing & splitting

import pandas as pd
import numpy as np

#LabelEncoder is used to convert categorical (text) data into numerical values
#StandardScaler is used to scale numerical data
#It converts data to have:
#Mean = 0
#Standard deviation = 1
#This helps ML models perform better.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# STEP 2: Load the Dataset

df = pd.read_csv(r"C:\Users\Samsung\Downloads\churn-bigml-80.csv")

# Display first 5 rows to confirm dataset loaded correctly
print("First 5 rows of the dataset:")
print(df.head())


# STEP 3: Understand the Dataset
# info() shows column names, data types, and missing values
print("\nDataset Information:")
print(df.info())

# Check total missing values in each column
print("\nMissing values in each column:")
print(df.isnull().sum())


# STEP 4: Handle Missing Values
# Numerical columns -> fill with MEAN
# select_dtypes → pick number columns
# fillna(mean) → fill empty values with average of that column .
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Categorical columns -> fill with MODE
# This code selects all categorical (text) columns from the DataFrame.
# It fills missing values (NaN) in each text column with the most frequent value (mode) of that column.

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verify missing values are handled
# “Show how many missing values are left in each column after data cleaning”
print("\nMissing values after handling:")
print(df.isnull().sum())


# STEP 5: Encode Categorical Variables
# Convert text data into numeric form using Label Encoding
# This code converts categorical (text) columns into numeric values using LabelEncoder.
# It prints a confirmation message after encoding is completed successfully.

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
print("\nCategorical columns encoded successfully.")


# STEP 6: Feature Scaling
# Standardize numerical features (important for ML models)
# This code scales all numeric columns in the DataFrame using StandardScaler.
#It ensures numeric features have mean = 0 and standard deviation = 1, improving model performance.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("\nNumerical features scaled using StandardScaler.")


# STEP 7: Split Features and Target
# Target variable is assumed as 'Churn'
# axis=0 → row, axis=1 → column
# drop() doesn’t change the original DataFrame unless inplace=True
# drop() deletes the column from the dataset
X = df.drop('Churn', axis=1)   # Features
y = df['Churn']                # Target
print("\nFeatures and target separated.")


# STEP 8: Train-Test Split
# 80% training data and 20% testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData split completed.")
print("Training data shape:", X_train.shape) # Shape --> Dimension
print("Testing data shape:", X_test.shape)


# STEP 9: Final Output
print("\n TASK 1 DATA PREPROCESSING COMPLETED SUCCESSFULLY!!!")
