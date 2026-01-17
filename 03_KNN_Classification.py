# STEP 1: Import required libraries
# pandas is used for handling and analyzing datasets
# numpy is used for numerical and mathematical operations
# matplotlib is used for data visualization
# sklearn provides machine learning algorithms and utilities

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# STEP 2: Load the dataset
# The Iris dataset is a built-in dataset in sklearn
# It contains 150 samples of flowers with 4 features each:
# sepal length, sepal width, petal length, and petal width
# The target variable represents three different flower species

data = load_iris()


# STEP 3: Separate features and target labels
# X stores the input features (measurements of flowers)
# y stores the output labels (flower species)
# This separation is necessary for training a supervised model

X = data.data
y = data.target


# STEP 4: Convert data into DataFrame (optional)
# Converting data into a DataFrame helps in understanding
# the structure of the dataset and viewing column names

df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y


# STEP 5: Split dataset into training and testing sets
# Training data is used to train the KNN model
# Testing data is used to evaluate how well the model performs
# test_size=0.2 means 20% data is used for testing
# random_state ensures the same split every time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# STEP 6: Create the KNN classifier
# KNN works by finding the K nearest data points
# n_neighbors defines the value of K
# Here, K is set to 5

knn = KNeighborsClassifier(n_neighbors=5)


# STEP 7: Train the KNN model
# The model learns by storing the training data
# KNN does not build a model explicitly, it memorizes data

knn.fit(X_train, y_train)

# STEP 8: Make predictions
# The trained model predicts the class labels
# for the test dataset based on nearest neighbors

y_pred = knn.predict(X_test)

# STEP 9: Evaluate the model
# Accuracy shows the percentage of correct predictions
# Confusion matrix shows correct and incorrect classifications
# Classification report provides precision, recall, and F1-score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# STEP 10: Test different values of K
# Different values of K are tested to find the optimal value
# Accuracy for each K is stored and plotted

accuracy_scores = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# STEP 11: Plot K vs Accuracy graph
# This graph helps in selecting the best K value
# The K with highest accuracy is considered optimal

plt.plot(range(1, 21), accuracy_scores)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.title("KNN: K Value vs Accuracy")
plt.show()
