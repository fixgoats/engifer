import os
from time import strftime, gmtime

import torch
import numpy as np
from numpy.fft import fft, fftshift
from numba import vectorize, float64, complex128
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from src.solvers import SsfmGPGPU, npnormSqr
from pathlib import Path

now = gmtime()
day = strftime('%Y-%m-%d', now)
timeofday = strftime('%H.%M', now)
basedir = os.path.join('graphs', day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)

end = 30
start = -30
samples = 512
dt = 0.1
dx = (end - start) / samples
kmax = np.pi / dx
dk = 2 * kmax / samples
pump = 20
sigma = 2

@vectorize([float64(complex128)])
def norm_sqr(x):
    return x.real * x.real + x.imag * x.imag


def gauss(x, y, sigmax, sigmay):
    return torch.exp(-x * x / sigmax - y * y / sigmay)


def npgauss(x, y):
    return np.exp(-x * x - y * y)


def fftangmom(psi, xv, yv, kxv, kyv):
    psiykx = fftshift(fft(fftshift(psi, axes=0), axis=0), axes=0)
    psixky = fftshift(fft(fftshift(psi, axes=1), axis=1), axes=1)
    return np.sum(norm_sqr(psixky) * xv * kyv - norm_sqr(psiykx) * yv * kxv) / np.shape(psi)[0]


def figBoilerplate():
    plt.cla()
    fig, ax = plt.subplots()
    fig.dpi = 300
    fig.figsize = (6.4, 3.6)
    return fig, ax


def smoothnoise(xv, yv):
    random = np.random.uniform(-1, 1, np.shape(xv)) + 1j * np.random.uniform(-1, 1, np.shape(xv))
    krange = np.linspace(-2, 2, num=21)
    kbasex, kbasey = np.meshgrid(krange, krange)
    kernel = npgauss(kbasex, kbasey)
    kernel /= np.sum(kernel)
    output = convolve2d(random, kernel, mode='same')
    return output / np.sqrt(np.sum(norm_sqr(output)))


siminfo = f'p{pump}'\
       f'n{samples}'\
       f's{sigma}dt{dt}'\
       f'xy{end-start}full'


def imshowBoilerplate(data, filename, xlabel, ylabel, extent, title=""):
    fig, ax = figBoilerplate()
    im = ax.imshow(data,
                   aspect='auto',
                   origin='lower',
                   interpolation='none',
                   extent=extent)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    name = f'{filename}{siminfo}'
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(f'{basedir}/{name}.pdf')
    plt.close()
    print(f'Made plot {name} in {basedir}')



cuda = torch.device('cuda')
# x = torch.arange(start, end, dx).type(dtype=torch.cfloat)
# xv, yv = torch.meshgrid(x, x, indexing='ij')
x = np.arange(start, end, dx)
k = np.arange(-kmax, kmax, dk)
xv, yv = np.meshgrid(x, x)
kxv, kyv = np.meshgrid(k, k)
bleh = smoothnoise(xv, yv)
print(fftangmom(bleh, xv, yv, kxv, kyv))
print(np.sum(norm_sqr(bleh)))
