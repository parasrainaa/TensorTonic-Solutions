import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    n = len(A)
    m = len(A[0])
    res = np.zeros((m,n))
    for i in range(n):
        for j in range(m):
                res[j][i] = A[i][j]
    return res