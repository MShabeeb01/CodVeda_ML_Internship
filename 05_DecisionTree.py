
# 1. Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 2. Load the Iris dataset (inbuilt dataset)
iris = load_iris()

# Convert to DataFrame for better understanding
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 3. Split the dataset into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Decision Tree Classifier (without pruning)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 5. Visualize the Decision Tree
plt.figure(figsize=(15, 8))
plot_tree(
    dt_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Decision Tree - Iris Dataset")
plt.show()

# 6. Make predictions on test data
y_pred = dt_model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Prune the Decision Tree to prevent overfitting
# Limiting tree depth
pruned_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
pruned_dt.fit(X_train, y_train)

# 9. Visualize the pruned tree
plt.figure(figsize=(15, 8))
plot_tree(
    pruned_dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Pruned Decision Tree (max_depth = 3)")
plt.show()

# 10. Evaluate pruned model
y_pred_pruned = pruned_dt.predict(X_test)

accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
f1_pruned = f1_score(y_test, y_pred_pruned, average='weighted')

print("Pruned Tree Accuracy:", accuracy_pruned)
print("Pruned Tree F1 Score:", f1_pruned)



# =====================================================
# DECISION TREE CLASSIFICATION
# =====================================================

# Decision Tree is a supervised machine learning algorithm
# used for classification and regression problems.
# It works by repeatedly splitting the dataset into smaller
# subsets based on the most important features.

# =====================================================
# DATASET USED: IRIS DATASET
# =====================================================

# The Iris dataset is a multiclass classification dataset.
# It contains 150 samples of flowers belonging to 3 species:
# - Setosa
# - Versicolor
# - Virginica

# Input features in the dataset are:
# - Sepal length
# - Sepal width
# - Petal length
# - Petal width

# The target variable is the flower species.

# =====================================================
# STEP 1: DATA LOADING
# =====================================================

# The dataset is loaded from an inbuilt source.
# Feature values are separated as input variables (X).
# Species labels are stored as the target variable (y).

# =====================================================
# STEP 2: TRAIN–TEST SPLITTING
# =====================================================

# The dataset is divided into training and testing sets.
# Training data is used to train the decision tree model.
# Testing data is used to evaluate the model performance.

# Common split ratio:
# - 80% training data
# - 20% testing data

# Random state is used to ensure reproducibility of results.

# =====================================================
# STEP 3: TRAINING THE DECISION TREE MODEL
# =====================================================

# The Decision Tree classifier learns decision rules
# based on feature values.

# It selects the best feature for splitting using
# impurity measures such as:
# - Gini Index
# - Entropy

# The model continues splitting until:
# - All data points are classified correctly, or
# - Stopping conditions are reached.

# =====================================================
# STEP 4: DECISION TREE VISUALIZATION
# =====================================================

# The trained decision tree can be visualized as a tree diagram.
# Each node shows:
# - Feature used for splitting
# - Threshold value
# - Gini impurity
# - Number of samples
# - Class distribution

# Visualization helps in understanding how decisions are made.

# =====================================================
# STEP 5: MAKING PREDICTIONS
# =====================================================

# The trained model predicts class labels
# for the unseen test data.

# These predicted labels are compared
# with the actual labels to measure performance.

# =====================================================
# STEP 6: MODEL EVALUATION
# =====================================================

# Accuracy:
# Measures the proportion of correctly classified instances.

# F1-Score:
# Harmonic mean of precision and recall.
# Useful when class distribution is uneven.

# Classification report:
# Displays precision, recall, F1-score
# and support for each class.

# =====================================================
# STEP 7: OVERFITTING IN DECISION TREES
# =====================================================

# Decision Trees tend to overfit the training data
# by creating very deep trees.

# Overfitting reduces performance on unseen data.

# =====================================================
# STEP 8: PRUNING THE DECISION TREE
# =====================================================

# Pruning is used to control tree growth
# and reduce overfitting.

# One common pruning method is limiting tree depth.
# This prevents the model from learning unnecessary details.

# A pruned tree is simpler, faster,
# and generalizes better to new data.

# =====================================================
# STEP 9: EVALUATION AFTER PRUNING
# =====================================================

# The pruned model is evaluated again
# using accuracy and F1-score.

# Usually, pruned trees show similar
# or better performance on test data.

# CONCLUSION

# Decision Trees are easy to interpret and visualize.
# They work well for both binary and multiclass classification.
# However, pruning is essential to avoid overfitting
# and improve generalization.
