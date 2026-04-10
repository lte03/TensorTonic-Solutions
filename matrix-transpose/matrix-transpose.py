import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A_ =  np.asarray(A)
    m,n = A_.shape
    transpose = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            transpose[i][j] = A_[j][i]
    return transpose
