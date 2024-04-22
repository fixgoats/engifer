import math

import numpy as np
import torch
import torch.fft as tfft
from numba import complex64, float32, vectorize
from scipy.signal import convolve2d

constV = -0.5j * 0.2
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 24
sigma = 2
dt = 0.05
Gamma = 0.1
Gammainv = 1.0 / Gamma
pump = 0
eta = 0
constpart = 0
dt = 0.05


def gauss(x, y, a=1, scale=1):
    return scale * np.exp(-(x * x + y * y) / a)


@vectorize([float32(complex64)])
def npnormSqr(x):
    return x.real * x.real + x.imag * x.imag


def smoothnoise(xv, yv):
    random = np.random.uniform(-1, 1, np.shape(xv)) + 1j * np.random.uniform(
        -1, 1, np.shape(xv)
    )
    krange = np.linspace(-2, 2, num=21)
    kbasex, kbasey = np.meshgrid(krange, krange)
    kernel = gauss(kbasex, kbasey)
    kernel /= np.sum(kernel)
    output = convolve2d(random, kernel, mode="same")
    output = convolve2d(output, kernel, mode="same")
    return output / np.sqrt(np.sum(npnormSqr(output)))

startX = 
xv = np.arange(startX, endX, dx)
psi = torch.from_numpy(smoothnoise())
psik = torch.tensor([[]])
nR = torch.tensor([[]])
kTimeEvo = torch.tensor([[]])


@torch.jit.script
def V(psi, nR):
    return (
        constpart + alpha * (psi.conj() * psi()) + (G + 0.5j * R) * nR + 0.5j * R * nR
    )


@torch.jit.script
def halfRStepPsi(psi, nR):
    return psi * torch.exp(-0.5j * dt * V(psi, nR))


@torch.jit.script
def tnormSqr(x):
    return x.conj() * x


@torch.jit.script
def halfRStepNR(psi, nR):
    return (
        math.exp(-0.5 * dt * Gamma) * torch.exp((-0.5 * dt * R) * tnormSqr(psi)) * nR
        + pump * dt * 0.5
    )


@torch.jit.script
def step(psi0, nR0):
    nR = halfRStepPsi(psi0, nR0)
    psik = tfft.fft2(psi0)
    psik *= kTimeEvo
    psi = tfft.ifft2(psik)
    nR = halfRStepNR(psi, nR)
    psi = halfRStepPsi(psi, nR)
    nR = halfRStepNR(psi, nR)
    return psi, nR
