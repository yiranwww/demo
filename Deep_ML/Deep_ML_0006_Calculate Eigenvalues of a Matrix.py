# Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.

# Example:
# Input:
# matrix = [[2, 1], [1, 2]]
# Output:
# [3.0, 1.0]
# Reasoning:
# The eigenvalues of the matrix are calculated using the characteristic equation of the matrix, which for a 2x2 matrix is 
# λ ^2 - trace(A)λ + det(A) where λ are the eigenvalues
import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	matrix_np = np.array(matrix)
	eigenvalues, eigenvectors = np.linalg.eig(matrix_np)
	
	return eigenvalues


# test case
matrix = [[2, 1], [1, 2]]
res = calculate_eigenvalues(matrix)
print(res)  # Expected output: [3.0, 1.0]s