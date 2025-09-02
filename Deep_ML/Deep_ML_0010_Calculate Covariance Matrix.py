# Write a Python function to calculate the covariance matrix for a given set of vectors. 
# The function should take a list of lists, where each inner list represents a feature with its observations, 
# and return a covariance matrix as a list of lists. 
# Additionally, provide test cases to verify the correctness of your implementation.
import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	n_features, n_observations = len(vectors), len(vectors[0])
	# mean of each feature
	means = [sum(observations)/n_observations for observations in vectors]
	cov_matrix = [[0] * n_features for _ in range(n_features)]
	# calculate the covariance for each pair
	for i in range(n_features):
		for j in range(i, n_features):
			mean_i, mean_j = means[i], means[j]
			cov_ij = sum((obs_i - mean_i) * (obs_j-mean_j) for obs_i, obs_j in zip(vectors[i], vectors[j]))
			cov_matrix[i][j] = cov_ij / (n_observations -1) 
			cov_matrix[j][i] = cov_matrix[i][j]  # covariance matrix is symmetric
	return cov_matrix 


    # cov_matrix = np.cov(vectors)
    # return cov_matrix.tolist()


# test case
input = [[1, 2, 3], [4, 5, 6]]
res = calculate_covariance_matrix(input)
print(res)  # Expected output: [[1.0, 1.0], [1.0, 1.0]]