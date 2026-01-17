import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# 1. Load the labeled dataset
df = pd.read_csv(r"C:\Users\Samsung\Downloads\Social_Network_Ads.csv")
X = df.iloc[:, [2, 3]].values  # Age and EstimatedSalary
y = df.iloc[:, 4].values       # Purchased (0 or 1)

# 2. Split and Scale (Crucial for SVM)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 3. Compare Kernels: Linear vs RBF
kernels = ['linear', 'rbf']
models = {}

for kernel in kernels:
    print(f"\n--- Results for {kernel.upper()} Kernel ---")
    classifier = SVC(kernel=kernel, probability=True, random_state=42)
    classifier.fit(X_train, y_train)
    models[kernel] = classifier
    
    # Make Predictions
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    
    # 4. Evaluate Metrics (Objective: Accuracy, Precision, Recall, AUC)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# 5. Visualize the Decision Boundary (Objective)
def plot_boundary(model, title, ax):
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    ax.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap='coolwarm')
    for i, j in enumerate(np.unique(y_set)):
        ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], label=j, edgecolors='k')
    ax.set_title(title)
    ax.legend()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_boundary(models['linear'], 'SVM Decision Boundary (Linear)', ax1)
plot_boundary(models['rbf'], 'SVM Decision Boundary (RBF)', ax2)
plt.tight_layout()
plt.show()


# ==============================================================================
# TASK 2: SUPPORT VECTOR MACHINE (SVM) FOR BINARY CLASSIFICATION - NOTES
# ==============================================================================

# 1. GOAL OF THE TASK
# ------------------
# To build a model that can predict which of two classes a data point belongs to.
# The SVM does this by finding a "Hyperplane" (a boundary line) that separates
# the classes with the largest possible gap (the margin).

# 2. KEY CONCEPTS FOR BEGINNERS
# -----------------------------
# * SUPPORT VECTORS: These are the data points closest to the boundary. They are
#   the most important points because the model's boundary is built based on them.
# * MARGIN: This is the "empty street" between the two classes. SVM always tries
#   to make this street as wide as possible to avoid mistakes on new data.
# * KERNEL TRICK: This allows the SVM to handle complex data. If a straight line
#   cannot separate the classes, the kernel "lifts" the data into a higher 
#   dimension where a separation is possible.

# 3. UNDERSTANDING KERNELS (OBJECTIVE COMPARISON)
# -----------------------------------------------
# * LINEAR KERNEL: 
#   - Best for data that is "Linearly Separable" (can be split by a straight line).
#   - Faster to train but less flexible for complex patterns.
# * RBF (RADIAL BASIS FUNCTION) KERNEL:
#   - The most popular kernel. It can create circular or curved boundaries.
#   - Best for real-world data where classes often overlap or mix.

# 

# 4. PRE-PROCESSING REQUIREMENTS
# ------------------------------
# * FEATURE SCALING: This is MANDATORY for SVM. Since SVM calculates distances
#   between points, all features (like Age vs. Salary) must be on the same scale.
#   Without scaling, the feature with the largest numbers will dominate the model.
# * TRAIN/TEST SPLIT: We train on one set and test on another to ensure the 
#   model isn't just "memorizing" the answers (Overfitting).

# 5. PERFORMANCE METRICS (OBJECTIVE EVALUATION)
# ---------------------------------------------
# To prove the model is successful, we measure:
# * ACCURACY: How many total predictions were correct?
# * PRECISION: Of the positive predictions, how many were actually correct?
# * RECALL: How many of the actual positive cases did we correctly identify?
# * AUC (Area Under Curve): Measures the model's ability to distinguish between 
#   the two classes across different thresholds. 1.0 is perfect; 0.5 is random.

# 6. VISUALIZATION
# ----------------
# * DECISION BOUNDARY: A plot that colors the "Decision Regions." It helps us 
#   see where the model thinks "Class A" ends and "Class B" begins.
# * If the boundary is a straight line, it's a Linear SVM. 
# * If the boundary is curved or has islands, it's an RBF SVM.

# ==============================================================================
# TOOLS USED: Python, Scikit-Learn, Pandas, Matplotlib
# ==============================================================================
