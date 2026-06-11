# 步骤
# 1. 随机选择k个中心
# 2. 每个点分配到最近中心
# 3. 更新中心 = 当前簇的均值
# 4. 重复直到收敛

import numpy as np

class KMeans:
    def __init__(self, k = 4, max_iters = 100, tol = 1e-4, random_state = 42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
    
    def fit(self, X):
        np.random.seed(self.random_state)

        # 1. 初始化中心点，随机选择k个样本
        idx = np.random.choice(X.shape[0], self.k, replace = False)
        self.centroids = X[idx]

        for i in range(self.max_iters):
            # 2. 计算距离 N,k
            distances = self._compute_distances(X)
            # 3. 分配簇
            labels = np.argmin(distances, axis = 1)
            # 4. 更新中心点
            new_centroids = np.array([
                X[labels == j].mean(axis = 0) if np.sum(labels==j)> 0 else self.centroids[j] for j in range(self.k)])
            # 5. 检查收敛
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break
        
        self.lables_ = labels
        return self
    
    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis = 1)

    def _compute_distances(self, X):
        return np.linalg.norm(X[:, None] - self.centroids[None, :], axis = 2)


# Test 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples = 300, centers = 4, cluster_std = 1.0, random_state = 0)
kmeans = KMeans(k = 4)
kmeans.fit(X)

labels = kmeans.lables_
centroids = kmeans.centroids

plt.scatter(X[:, 0], X[:, 1], c = labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c = 'red', marker = 'X', s=200)
plt.show()