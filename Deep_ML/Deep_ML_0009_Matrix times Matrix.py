# multiply two matrices together (return -1 if shapes of matrix dont aline), i.e. C = A dot B 
import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	a = np.array(a)
	b = np.array(b)
	
	m_a, n_a = len(a), len(a[0])
	n_b, a_b = len(b), len(b[0])
	if n_a != n_b:
		return -1
	n = n_a
	c = np.zeros((m_a, a_b))
	for i in range(m_a):
		for j in range(a_b):
			for k in range(n):
				c[i][j] += a[i][k] * b[k][j]
	return c

# test case
A = [[1,2],[2,4]]
B = [[2,1],[3,4]]
res = matrixmul(A,B)
print(res)  # Expected output: [[8. 9.] [18. 18.]]