import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	b = np.array(a).T
	return b





def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return [list(i) for i in zip(*a)]

