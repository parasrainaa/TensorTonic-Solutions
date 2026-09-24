import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    res = 0.00
    n = len(A)
    m = len(A[0])
    for i in range(n):
        for j in range(m):
            if i == j:
                res += A[i][j]
    return res
                