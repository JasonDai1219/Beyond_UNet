import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class BiKMeans:
    def __init__(self, n_clusters, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit(self, X):
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))
        self._fit_recursive(X, 0, self.n_clusters)

    def _fit_recursive(self, X, start_idx, end_idx):
        if end_idx - start_idx <= 1:
            return

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=2, n_init=10, max_iter=self.max_iterations)
        kmeans.fit(X)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        # Assign the cluster centers to the correct slice of self.cluster_centers_
        mid_idx = start_idx + (end_idx - start_idx) // 2
        self.cluster_centers_[start_idx:mid_idx] = cluster_centers[0]  # Cluster 0
        self.cluster_centers_[mid_idx:end_idx] = cluster_centers[1]  # Cluster 1

        # Recursively apply Bi-Kmeans to the sub-clusters
        self._fit_recursive(X[labels == 0], start_idx, start_idx + np.sum(labels == 0))
        self._fit_recursive(X[labels == 1], end_idx - np.sum(labels == 1), end_idx)

    def predict(self, X):
        # Predict labels for X
        distances = pairwise_distances_argmin_min(X, self.cluster_centers_)[0]
        return distances, self.cluster_centers_
    

def calculate_within_cluster_distance(X, labels, cluster_centers):
    # X = torch.tensor(X, dtype=torch.float32)
    # labels = torch.tensor(labels, dtype=torch.long)
    # cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32)
    
    distances = torch.zeros(len(X), dtype=torch.float32).cuda()
    for i in range(len(cluster_centers)):
        cluster_points = X[labels == i]
        distances[labels == i] = torch.norm(cluster_points - cluster_centers[i], dim=1)
    
    return torch.mean(distances)

def calculate_between_cluster_distance(cluster_centers):
    # cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32)
    num_clusters = len(cluster_centers)
    distances = torch.zeros((num_clusters, num_clusters), dtype=torch.float32).cuda()
    
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                distances[i, j] = torch.norm(cluster_centers[i] - cluster_centers[j])
    
    return torch.mean(distances)

def clustering_loss(X, labels, cluster_centers, alpha=1.0, beta=1.0):
    within_cluster_distance = calculate_within_cluster_distance(X, labels, cluster_centers)
    between_cluster_distance = calculate_between_cluster_distance(cluster_centers)
    
    within_cluster_distance += 1e-6  # Prevent division by zero
    loss = alpha * (within_cluster_distance / (between_cluster_distance + 1e-6))
    #loss = alpha * within_cluster_distance - beta * between_cluster_distance
    
    return loss