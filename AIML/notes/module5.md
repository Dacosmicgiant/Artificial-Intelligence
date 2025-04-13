# Unsupervised Machine Learning: Clustering Techniques

Unsupervised learning is a branch of machine learning that deals with finding patterns or structures in unlabeled data. Among various unsupervised learning techniques, clustering is one of the most popular and widely used approaches.

## What is Clustering?

Clustering is the task of dividing data points into groups (clusters) such that:
- Data points within the same cluster are similar to each other
- Data points in different clusters are dissimilar from each other

The similarity or dissimilarity is usually measured using a distance function like Euclidean distance, Manhattan distance, or cosine similarity.

## Types of Clustering Algorithms

### 1. K-Means Clustering

K-means is one of the simplest and most popular clustering algorithms. It partitions data into K clusters, where each data point belongs to the cluster with the nearest mean.

**Algorithm:**
1. Select K points as initial centroids
2. Assign each data point to the closest centroid
3. Recalculate centroids based on the assigned points
4. Repeat steps 2-3 until centroids no longer move significantly

**Python Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-means
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot results
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'y']
for i in range(len(X)):
    plt.scatter(X[i, 0], X[i, 1], color=colors[labels[i]], alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.7, marker='x')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

**Advantages:**
- Simple and easy to implement
- Efficient for large datasets
- Works well when clusters are spherical and similar in size

**Disadvantages:**
- Requires specifying K in advance
- Sensitive to initial centroid positions
- Not suitable for non-spherical clusters
- Can converge to local optima

### 2. K-Medoids (PAM - Partitioning Around Medoids)

K-medoids is similar to K-means but uses actual data points as cluster centers (medoids) rather than means, making it more robust to outliers.

**Algorithm:**
1. Select K data points as initial medoids
2. Assign each data point to the closest medoid
3. For each cluster, find the point that minimizes the sum of distances to other points
4. Replace the current medoid with this new point if it reduces the overall cost
5. Repeat steps 2-4 until medoids don't change

**Python Implementation:**

```python
import numpy as np
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-medoids
kmedoids = KMedoids(n_clusters=4, random_state=0).fit(X)
labels = kmedoids.labels_
medoids = X[kmedoids.medoid_indices_]

# Plot results
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'y']
for i in range(len(X)):
    plt.scatter(X[i, 0], X[i, 1], color=colors[labels[i]], alpha=0.5)

plt.scatter(medoids[:, 0], medoids[:, 1], c='black', s=200, alpha=0.7, marker='X')
plt.title('K-medoids Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

**Advantages:**
- More robust to outliers than K-means
- Can work with arbitrary distance metrics
- Cluster centers are actual data points

**Disadvantages:**
- More computationally expensive than K-means
- Still requires specifying K in advance
- Not effective for non-spherical clusters

### 3. Hierarchical Clustering

Hierarchical clustering builds a tree of clusters (dendrogram) by either merging smaller clusters into larger ones (agglomerative) or splitting larger clusters into smaller ones (divisive).

#### 3.1 Agglomerative Clustering

In agglomerative clustering, each data point starts as its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

**Algorithm:**
1. Start with each data point as a single cluster
2. Compute pairwise distances between clusters
3. Merge the two closest clusters
4. Update distances between the new cluster and remaining clusters
5. Repeat steps 3-4 until only one cluster remains

**Linkage Methods:**
- **Single linkage:** Distance between closest points
- **Complete linkage:** Distance between furthest points
- **Average linkage:** Average distance between all pairs of points
- **Ward linkage:** Minimizes the increase in variance after merging

**Python Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=100, centers=4, cluster_std=0.60, random_state=0)

# Create linkage matrix for dendrogram
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.tight_layout()
plt.show()

# Apply agglomerative clustering
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward').fit(X)
labels = agg_clustering.labels_

# Plot clusters
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'y']
for i in range(len(X)):
    plt.scatter(X[i, 0], X[i, 1], color=colors[labels[i]], alpha=0.5)

plt.title('Agglomerative Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

#### 3.2 Divisive Clustering

Divisive clustering works in the opposite direction of agglomerative clustering. It starts with all data points in one cluster and recursively splits them into smaller clusters.

**Algorithm:**
1. Start with all data points in a single cluster
2. Find the point that is furthest from all other points
3. Split the cluster based on points closer to either the original center or the new point
4. Recursively apply the algorithm to each subcluster
5. Stop when a termination condition is met

**Python Implementation:**
Divisive clustering isn't directly implemented in scikit-learn, but can be simulated using recursive K-means splitting:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def divisive_clustering(X, max_clusters=4, min_samples=5):
    """A simple implementation of divisive clustering using K-means for splitting"""
    # Start with all points in one cluster
    all_clusters = [np.arange(len(X))]
    all_centroids = [np.mean(X, axis=0)]
    
    # Continue splitting until we reach max_clusters
    while len(all_clusters) < max_clusters:
        max_diameter = -1
        cluster_to_split = -1
        
        # Find the cluster with the largest diameter
        for i, cluster in enumerate(all_clusters):
            if len(cluster) <= min_samples:
                continue
                
            # Compute diameter (max distance between any two points)
            cluster_points = X[cluster]
            dists = np.max(np.linalg.norm(
                cluster_points[:, np.newaxis] - cluster_points[np.newaxis, :], 
                axis=2
            ))
            
            if dists > max_diameter:
                max_diameter = dists
                cluster_to_split = i
        
        if cluster_to_split == -1:
            break
            
        # Split the selected cluster using K-means
        cluster_points = X[all_clusters[cluster_to_split]]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_points)
        
        # Create new clusters
        mask = kmeans.labels_ == 0
        cluster1 = all_clusters[cluster_to_split][mask]
        cluster2 = all_clusters[cluster_to_split][~mask]
        
        # Update clusters
        all_clusters.pop(cluster_to_split)
        all_clusters.extend([cluster1, cluster2])
        
        # Update centroids
        all_centroids.pop(cluster_to_split)
        all_centroids.extend([np.mean(X[cluster1], axis=0), np.mean(X[cluster2], axis=0)])
    
    # Create label array
    labels = np.zeros(len(X), dtype=int)
    for i, cluster in enumerate(all_clusters):
        labels[cluster] = i
    
    return labels, np.array(all_centroids)

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply divisive clustering
labels, centroids = divisive_clustering(X, max_clusters=4)

# Plot results
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'y']
for i in range(len(X)):
    plt.scatter(X[i, 0], X[i, 1], color=colors[labels[i]], alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.7, marker='x')
plt.title('Divisive Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

**Advantages of Hierarchical Clustering:**
- No need to specify the number of clusters in advance
- Provides a dendrogram visualization that can help choose the number of clusters
- Can uncover hierarchical relationships in the data
- Can handle various distance metrics

**Disadvantages:**
- Computationally expensive (O(n³) for standard implementations)
- No global objective function is optimized
- Cannot undo previous steps (greedy approach)

### 4. Density-Based Clustering: DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) defines clusters as dense regions separated by regions of lower density. It can find arbitrarily shaped clusters and is robust to outliers.

**Key Concepts:**
- **Core Point:** A point with at least `min_samples` points within a distance of `eps`
- **Border Point:** A point within `eps` distance of a core point but with fewer than `min_samples` points within `eps`
- **Noise Point:** A point that is neither a core point nor a border point

**Algorithm:**
1. For each point, determine if it's a core point
2. Connect core points that are within `eps` distance of each other
3. Form clusters from connected core points
4. Assign each border point to the cluster of its neighboring core point
5. Assign noise points to a separate "noise" cluster

**Python Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate sample data - DBSCAN works well with non-spherical clusters
X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

# Add some outliers
X = np.vstack([X, np.array([[3, 3], [-3, 3], [3, -3], [-3, -3]])])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5).fit(X)
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')

# Plot results
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:  # Black used for noise
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], color=col, alpha=0.6,
                s=50 if k != -1 else 20, label=f'Cluster {k}' if k != -1 else 'Noise')

plt.title(f'DBSCAN: {n_clusters} clusters and {n_noise} noise points')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

**Advantages:**
- Does not require specifying the number of clusters in advance
- Can find arbitrarily shaped clusters
- Robust to outliers
- Only requires two parameters: `eps` and `min_samples`

**Disadvantages:**
- Sensitive to parameter settings
- Not effective for datasets with varying densities
- Struggles with high-dimensional data due to the "curse of dimensionality"

## Comparison of Clustering Methods

| Algorithm | Shapes | Outliers | Scalability | Parameters | Hierarchical |
|-----------|--------|----------|-------------|------------|--------------|
| K-Means | Spherical | Sensitive | Good | K | No |
| K-Medoids | Spherical | Robust | Moderate | K | No |
| Agglomerative | Various | Depends on linkage | Poor | Distance metric, linkage | Yes |
| Divisive | Various | Depends on splitting | Poor | Termination criteria | Yes |
| DBSCAN | Arbitrary | Robust | Good | eps, min_samples | No |

## Choosing the Right Clustering Algorithm

The choice of clustering algorithm depends on:

1. **Data characteristics**
   - Size and dimensionality of the dataset
   - Expected cluster shapes
   - Presence of outliers

2. **Algorithm properties**
   - Speed and scalability
   - Interpretability of results
   - Need to specify number of clusters in advance

3. **Application requirements**
   - Need for hierarchical structure
   - Need for probabilistic assignments
   - Need for deterministic results

## Evaluating Clustering Results

Without ground truth labels, clustering performance can be evaluated using:

1. **Internal metrics**
   - Silhouette score
   - Davies-Bouldin index
   - Calinski-Harabasz index

2. **Visual inspection**
   - Dimensionality reduction (PCA, t-SNE)
   - Dendrogram analysis (for hierarchical methods)

When ground truth is available, metrics like Adjusted Rand Index or Normalized Mutual Information can be used.

## Conclusion

Clustering is a powerful approach for unsupervised learning with many applications across domains. The choice of algorithm should be guided by the specific characteristics of your data and the requirements of your application. Often, it's beneficial to try multiple algorithms and compare their results.
