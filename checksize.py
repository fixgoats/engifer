import numpy as np
import numpy.linalg as la
import cmath as cm
from numba import njit


@njit
def sqNorm(x):
    return x.real * x.real + x.imag * x.imag

points = np.load("monotilegrid.npy")
distances = [np.sqrt(sqNorm(p2 - p1)) for p1 in points for p2 in points if p1 != p2]
print(np.min(distances))
