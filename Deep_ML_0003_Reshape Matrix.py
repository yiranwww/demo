import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    m, n = len(a), len(a[0])
    if m * n != new_shape[0] * new_shape[1]:
        return []
    a_np = np.array(a)
    reshaped_matrix = a_np.reshape(new_shape)
	return reshaped_matrix.tolist()
