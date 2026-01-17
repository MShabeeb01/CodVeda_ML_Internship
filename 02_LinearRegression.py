# Simple Linear Regression using Housing Data

# Step 1: Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 2: Load the CSV file using correct Windows path
# Use raw string (r"") to avoid path errors

data = pd.read_csv(r"C:\Users\Samsung\Downloads\Housing.csv")

# Display first few rows to confirm loading
print("Dataset Loaded Successfully!")
print(data.head())


# Step 3: Check dataset information
print("\nDataset Info:")
print(data.info())


# Step 4: Handle categorical (string) columns
# Convert categorical columns into numerical using one-hot encoding

data_encoded = pd.get_dummies(data, drop_first=True)


# Step 5: Separate input features (X) and target variable (y)
# Target column is usually 'price' in Housing dataset

X = data_encoded.drop("price", axis=1)   # input features
y = data_encoded["price"]                # target variable


# Step 6: Split data into training and testing sets
# 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 7: Create Linear Regression model
model = LinearRegression()


# Step 8: Train the model
model.fit(X_train, y_train)


# Step 9: Predict house prices
y_pred = model.predict(X_test)


# Step 10: Evaluate the model

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error (MSE):", mse)

# R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared (R²):", r2)


# Step 11: Display model coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients)


# Step 12: Display model intercept
print("\nModel Intercept:")
print(model.intercept_)
