# Write a Python function that multiplies a matrix by a scalar and returns the result.

# Example:
# Input:
# matrix = [[1, 2], [3, 4]], scalar = 2
# Output:
# [[2, 4], [6, 8]]
# Reasoning:
# Each element of the matrix is multiplied by the scalar.
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	m, n= len(matrix), len(matrix[0])
	result = [[0] * n for _ in range(m)]
	for i in range(m):
		for j in range(n):
			result[i][j] = matrix[i][j] * scalar
	return result

# test case
matrix = [[1, 2], [3, 4]]
scalar = 2
# output = [[2, 4], [6, 8]]

res = scalar_multiply(matrix, scalar)
print(res)  # Expected output: [[2, 4], [6, 8]]