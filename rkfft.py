from scipy.integrate import RK45
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import cm, animation
# import argparse


@jit(nopython=True)
def gauss(x, y):
    return np.exp(-x * x - y * y)


@jit(nopython=True)
def sqr_mod(x):
    return x.real * x.real + x.imag * x.imag


@jit(nopython=True)
def exactsol(x, y, t, a, m):
    return (a / (a + 1.0j * t / m)) \
        * np.exp(-(x * x + y * y) / (2 * (a + 1.0j * t / m)))


startx = -5
endx = -5
samples = 256
dx = (endx - startx) / samples
x = np.arange(startx, endx, dx)
xv, yv = np.meshgrid(x, x)
k0x = np.pi / dx
dkx = 2 * k0x / samples
kx = np.fft.fftshift(np.arange(-k0x, k0x, dkx))
kxv, kxy = np.meshgrid(kx, kx)


@jit(nopython=True)
def f(t, y):
    tmp = np.fft.fft2(np.reshape(y, (samples, samples)))
    tmp = -1.0j*(kxv * kxv + kxy * kxy) * tmp
    return np.fft.ifft2(tmp)


psi0 = gauss(xv, yv).astype('complex')
sims = [psi0.flatten()]
