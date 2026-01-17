# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r"C:\Users\Samsung\Downloads\Mall_Customers.csv")

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal number of clusters
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure()
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")
plt.show()

# Apply K-Means with optimal k (usually k = 5)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure()
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set2')
plt.scatter(
    kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
    kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
    s=200,
    marker='X')
plt.title("Customer Segmentation using K-Means")
plt.show()

# Topic: Introduction to K-Means Clustering
# K-Means is an unsupervised machine learning algorithm
# It groups data points based on similarity
# No target variable is required for clustering

# Topic: Dataset Description
# The customer segmentation dataset contains customer details
# It includes income and spending behavior
# The dataset is unlabeled and suitable for clustering

# Topic: Feature Selection
# Only numerical features are used for K-Means
# Annual Income represents customer earning capacity
# Spending Score represents customer purchasing behavior
# These features help identify spending patterns

# Topic: Data Preprocessing
# K-Means works using distance calculations
# Features with different ranges can affect results
# Data scaling is applied to normalize values

# Topic: Elbow Method
# The elbow method is used to find the optimal number of clusters
# WCSS is calculated for different values of k
# The point where WCSS decreases slowly is selected

# Topic: Applying K-Means Algorithm
# K-Means is initialized with the optimal cluster count
# Customers are grouped based on similarity
# Each data point is assigned to a cluster

# Topic: Cluster Visualization
# Clusters are visualized using a scatter plot
# Different colors represent different customer groups
# Centroids indicate the center of each cluster

# Topic: Result Interpretation
# Each cluster represents a unique customer segment
# High income and high spending customers form one group
# Low income and low spending customers form another group

# Topic: Conclusion
# K-Means clustering helps understand customer behavior
# It supports effective customer segmentation
# Businesses can use this for targeted marketing
