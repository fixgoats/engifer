import numpy as np

gamma = 1


def cylbasis(r, theta):
    return r * np.exp(-(gamma**2) / 2 + 1j * theta)


def pxbasis(r, theta):
    return r * np.cos(theta) * np.exp(-gamma * r**2 / 2)


def pybasis(r, theta):
    return r * np.cos(theta) * np.exp(-gamma * r**2 / 2)


def pxbasisx(x, y):
    return x * np.exp(-gamma * (x**2 + y**2) / 2)


def pybasisx(x, y):
    return y * np.exp(-gamma * (x**2 + y**2) / 2)


def integrate(x, y, r):
    return np.heaviside(r - (x**2 + y**2))
