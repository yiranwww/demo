# Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.

# Example:
# Input:
# matrix = [[4, 7], [2, 6]]
# Output:
# [[0.6, -0.7], [-0.2, 0.4]]
# Reasoning:
# The inverse of a 2x2 matrix [a, b], [c, d] is given by (1/(ad-bc)) * [d, -b], [-c, a], provided ad-bc is not zero.

import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    # matrix_np = np.array(matrix)
    # a = matrix_np[0][0]
    # b = matrix_np[0][1]
    # c = matrix_np[1][0]
    # d = matrix_np[1][1]
    # numerator = 1/(a * d - b * c)
    # inverse = [[0] * 2 for _ in range(2)]
    # inverse[0][0] = round(d * numerator, 4)
    # inverse[0][1] = round(-b * numerator, 4)
    # inverse[1][0] = round(-c * numerator, 4)
    # inverse[1][1] = round(a * numerator, 4)

    # matrix_np = np.array(matrix)
    # inverse = np.linalg.inv(matrix_np)
    return inverse

matrix = [[4, 7], [2, 6]]
res = inverse_2x2(matrix)
print(res)