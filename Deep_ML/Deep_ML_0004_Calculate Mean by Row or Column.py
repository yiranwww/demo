def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    m, n = len(matrix), len(matrix[0])

    if mode == "column":
        means = [0] * n
        for i in range(n):
            cur_col_sum = 0
            for j in range(m):
                cur_col_sum += matrix[j][i]
            means[i] = round(cur_col_sum / m, 4)
    else:
        means = [0] * m
        for i in range(m):
            cur_row_sum = 0
            for j in range(n):
                cur_row_sum += matrix[i][j]
            means[i] = round(cur_row_sum / n, 4)

    return means





# test case
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mode = 'column'
# output = [4.0, 5.0, 6.0]

res = calculate_matrix_mean(matrix, mode)
print(res)  # Expected output: [4.0, 5.0, 6.0]