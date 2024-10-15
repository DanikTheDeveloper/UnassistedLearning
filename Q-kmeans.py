
# 0. Adjust parameters
NUM_CLUSTERS = 2       # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER = 5           # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris)

# 1. Import any other required libraries (e.g., numpy, scikit-learn)
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 2. Load the Iris dataset using scikit-learn's load_iris() function
data = load_iris()
x = data.data
y = data.target
features = data.features
print(f"Features: {features}")
print(f"First 10 Samples:\n{x[:10]}")

# 3. Implement K-Means Clustering
    # 3.1. Import KMeans from scikit-learn
from sklearn.cluster import KMeans
    # 3.2. Create an instance of KMeans with the specified number of clusters and max_iter
NUM_CLUSTERS = 2  # Change to 2, 3, or 4
MAX_ITER = 5  # Change to 5, 10, or 20

kmeans = KMeans(n_clusters=NUM_CLUSTERS, max_iter=MAX_ITER, random_state=42)
    # 3.3. Fit the KMeans model to the data X
kmeans.fit(x)
    # 3.4. Obtain the cluster labels
labels = kmeans.labels_

print(f"Cluster Labels:\n{labels}")
print(f"Cluster Centers:\n{kmeans.cluster_centers_}")

# 4. Visualize the Results
    # 4.1. Extract the features for visualization
FEATURE_X_INDEX = 2  # Petal length
FEATURE_Y_INDEX = 3  # Petal width

x_feature = x[:, FEATURE_X_INDEX]
y_feature = x[:, FEATURE_Y_INDEX]
    # 4.2. Create a scatter plot of x_feature vs y_feature, colored by the cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(
    x_feature, y_feature, 
    c=labels, cmap='viridis', 
    s=50, alpha=0.8
)
    # 4.3. Use different colors to represent different clusters
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, FEATURE_X_INDEX], centroids[:, FEATURE_Y_INDEX], 
    c='red', s=200, marker='X', edgecolors='k', 
    label='Centroids'
)

plt.xlabel(features[FEATURE_X_INDEX])
plt.ylabel(features[FEATURE_Y_INDEX])
plt.title(f'K-Means Clustering (n_clusters={NUM_CLUSTERS}, max_iter={MAX_ITER})')
plt.legend()
plt.grid(True)
plt.show()