# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score


# 2. Load Dataset
# Replace path with your file location
df = pd.read_csv(r"C:\Users\Samsung\Downloads\customer_churn (1).csv")

# View first few rows
print(df.head())


# 3. Data Preprocessing

# Drop unnecessary columns (example: customerID)
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convert target variable to binary if needed
# Yes -> 1, No -> 0
if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']


# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 5. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 6. Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)


# 7. Model Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# 8. Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# 9. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# 10. Model Coefficients (Interpretation)
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Odds_Ratio': np.exp(model.coef_[0])})

print(coefficients.sort_values(by='Odds_Ratio', ascending=False))



# =====================================================================
# LOGISTIC REGRESSION FOR CUSTOMER CHURN PREDICTION – DETAILED NOTES
# =====================================================================

# Logistic Regression is a MACHINE LEARNING algorithm used for
# CLASSIFICATION problems, not regression.
# Even though the name contains "regression", it is used to
# predict categories (classes).

# In this problem, the task is BINARY CLASSIFICATION because
# there are only TWO possible outputs:
# 1 → Customer will churn (leave the service)
# 0 → Customer will not churn (stay with the service)


# ---------------------------------------------------------------------
# WHAT IS CUSTOMER CHURN?
# ---------------------------------------------------------------------
# Customer churn means when a customer stops using a company’s service.
# Companies want to predict churn in advance so they can:
# - Offer discounts
# - Improve services
# - Retain customers


# ---------------------------------------------------------------------
# STEP 1: IMPORTING LIBRARIES
# ---------------------------------------------------------------------
# A library is a collection of pre-written code that helps us
# perform tasks easily without writing everything from scratch.

# pandas:
# - Used to read CSV files
# - Used to store data in tables (DataFrames)
# - Used for cleaning and preprocessing data

# numpy:
# - Used for numerical operations
# - Works with arrays and mathematical calculations

# matplotlib:
# - Used to draw graphs and plots
# - Helps visualize model performance

# scikit-learn:
# - A popular machine learning library
# - Provides tools for data preprocessing, model training,
#   and performance evaluation


# ---------------------------------------------------------------------
# STEP 2: LOADING THE DATASET
# ---------------------------------------------------------------------
# The dataset is stored in a CSV (Comma Separated Values) file.
# pandas reads this file and stores it in a DataFrame.

# A DataFrame looks like an Excel table:
# - Rows represent customers
# - Columns represent features (customer details)

# Viewing the first few rows helps understand:
# - Column names
# - Type of data (numbers or text)
# - Target variable (Churn)


# ---------------------------------------------------------------------
# STEP 3: UNDERSTANDING THE DATA
# ---------------------------------------------------------------------
# Each row represents ONE customer.
# Each column represents ONE feature, such as:
# - Gender
# - Tenure
# - Monthly charges
# - Contract type

# The target column is "Churn"
# It tells whether the customer left or stayed.


# ---------------------------------------------------------------------
# STEP 4: DATA PREPROCESSING
# ---------------------------------------------------------------------
# Data preprocessing is the MOST IMPORTANT step in machine learning.
# Raw data cannot be directly given to a model.

# Common problems in raw data:
# - Unnecessary columns
# - Text (categorical) data
# - Different scales of values

# These problems must be fixed before training the model.


# ---------------------------------------------------------------------
# REMOVING UNNECESSARY COLUMNS
# ---------------------------------------------------------------------
# Columns like customerID are only for identification.
# They do not affect churn prediction.
# Keeping such columns can confuse the model.
# Therefore, they are removed.


# ---------------------------------------------------------------------
# CONVERTING TARGET VARIABLE
# ---------------------------------------------------------------------
# Machine learning models work only with numbers.
# If the Churn column contains text values like:
# "Yes" and "No"
# they must be converted into numbers.

# Conversion used:
# Yes → 1 (customer churned)
# No  → 0 (customer did not churn)


# ---------------------------------------------------------------------
# HANDLING CATEGORICAL FEATURES
# ---------------------------------------------------------------------
# Many columns contain text values such as:
# - Male / Female
# - Yes / No
# - Contract types

# Models cannot understand text.
# Label Encoding converts text into numeric values.
# Example:
# Male → 0
# Female → 1

# This allows the model to process the data.


# ---------------------------------------------------------------------
# STEP 5: SEPARATING FEATURES AND TARGET
# ---------------------------------------------------------------------
# Features (X):
# - All input columns used to make predictions
# - Example: tenure, charges, contract type

# Target (y):
# - Output column we want to predict
# - In this case: Churn

# The model learns patterns from X to predict y.


# ---------------------------------------------------------------------
# STEP 6: TRAIN–TEST SPLIT
# ---------------------------------------------------------------------
# The dataset is divided into two parts:

# Training data:
# - Used to train (teach) the model
# - Usually 70–80% of the dataset

# Testing data:
# - Used to test the model’s performance
# - Usually 20–30% of the dataset

# This helps check how well the model performs on new, unseen data.


# ---------------------------------------------------------------------
# STEP 7: FEATURE SCALING
# ---------------------------------------------------------------------
# Different features have different ranges.
# Example:
# - Tenure may range from 1 to 72
# - Monthly charges may range from 20 to 120

# Large values can dominate small values.
# Feature scaling solves this problem.

# StandardScaler:
# - Converts data to a standard range
# - Mean = 0
# - Standard deviation = 1

# This helps the model learn efficiently.


# ---------------------------------------------------------------------
# STEP 8: LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------------------
# Logistic Regression uses a mathematical function
# called the SIGMOID function.

# The sigmoid function converts values into probabilities
# between 0 and 1.

# If probability ≥ 0.5 → Predict churn (1)
# If probability < 0.5 → Predict no churn (0)


# ---------------------------------------------------------------------
# STEP 9: MODEL TRAINING
# ---------------------------------------------------------------------
# During training:
# - The model learns how features affect churn
# - It adjusts internal weights (coefficients)
# - The goal is to minimize prediction error


# ---------------------------------------------------------------------
# STEP 10: MAKING PREDICTIONS
# ---------------------------------------------------------------------
# The trained model predicts churn for test data.
# Two types of outputs are produced:
# - Class prediction (0 or 1)
# - Probability score (used for ROC curve)


# ---------------------------------------------------------------------
# STEP 11: MODEL EVALUATION METRICS
# ---------------------------------------------------------------------
# Accuracy:
# - Percentage of total correct predictions

# Precision:
# - Out of all predicted churns,
#   how many were actually churns

# Recall:
# - Out of all actual churns,
#   how many were correctly identified

# Confusion Matrix:
# - Shows true positives, true negatives,
#   false positives, and false negatives


# ---------------------------------------------------------------------
# STEP 12: ROC CURVE AND AUC
# ---------------------------------------------------------------------
# ROC curve compares:
# - True Positive Rate
# - False Positive Rate

# AUC (Area Under Curve):
# - Measures overall model performance
# - Value ranges from 0 to 1
# - Higher value means better model


# ---------------------------------------------------------------------
# STEP 13: INTERPRETING THE MODEL
# ---------------------------------------------------------------------
# Logistic Regression provides coefficients for each feature.

# Positive coefficient:
# - Increases probability of churn

# Negative coefficient:
# - Decreases probability of churn

# Odds Ratio:
# - Shows how strongly a feature influences churn
# - Values greater than 1 increase churn likelihood
# - Values less than 1 decrease churn likelihood
