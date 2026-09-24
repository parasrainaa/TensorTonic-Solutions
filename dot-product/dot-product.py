import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    n = len(x)
    res = 0.00
    for i in range(n):
        res += x[i]*y[i]
    return res
    