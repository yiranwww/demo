# Write a Python function that transforms a given matrix A using the operation T^-1AS, where T and S are invertible matrices. 
# The function should first validate if the matrices T and S are invertible, and then perform the transformation. 
# In cases where there is no solution return -1

# Example:
# Input:
# A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]
# Output:
# [[0.5,1.5],[1.5,3.5]]

import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	# check if T and S are invertible
	if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
		return -1

	transformed_matrix = np.linalg.inv(T) @ np.array(A) @ np.array(S)
	return transformed_matrix





A = [[1, 2], [3, 4]]
T = [[2, 0], [0, 2]]
S = [[1, 1], [0, 1]]
res = transform_matrix(A, T, S)
print(res)  # Expected output: [[0.5, 1.5], [1.5, 3.5]]