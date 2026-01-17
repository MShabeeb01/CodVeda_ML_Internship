# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv(r"C:\Users\Samsung\Downloads\heart_cleveland_upload.csv")

# Display column names (for verification)
print("Dataset Columns:\n", df.columns)

# Automatically detect target column
target_col = None
for col in ["target", "output", "condition", "num"]:
    if col in df.columns:
        target_col = col
        break

# Safety check
if target_col is None:
    raise ValueError("Target column not found in dataset")

print("Using target column:", target_col)

# Split features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

# Grid Search with cross-validation
grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Train model
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Cross-validation score
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="f1")
print("Average Cross-validation F1-score:", cv_scores.mean())

# Feature importance
importances = best_model.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()
