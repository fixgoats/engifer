import math

import chime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as tfft
from numba import complex128, float64, vectorize
from scipy.signal import convolve2d
from torch.profiler import ProfilerActivity, profile, record_function

from src.penrose import makeSunGrid

gammalp = 0.2
constV = -0.5j * gammalp
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 24
sigma = 2
dt = 0.05
Gamma = 0.1
Gammainv = 1.0 / Gamma
eta = 0
dt = 0.05
hbar = 6.582119569e-1  # meV * ps
m = 0.32


def figBoilerplate():
    plt.cla()
    fig, ax = plt.subplots()
    fig.dpi = 300
    fig.figsize = (6.4, 4.8)
    return fig, ax


def imshowBoilerplate(
    data, filename, xlabel="", ylabel="", extent=[], title="", aspect="auto"
):
    fig, ax = figBoilerplate()
    im = ax.imshow(
        data, aspect=aspect, origin="lower", interpolation="none", extent=extent
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(f"{filename}.pdf")
    plt.close()
    print(f"Made testplot")


def gauss(x, y, a=1, scale=1):
    return scale * np.exp(-(x * x + y * y) / a)


def tgauss(x, y, s=1):
    return torch.exp(-((x / s) ** 2) - (y / s) ** 2)


@vectorize([float64(complex128)])
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


N = 1024
startX = -100
endX = 100
dx = (endX - startX) / N
x = np.arange(startX, endX, dx)
xv, yv = np.meshgrid(x, x)
psi = torch.from_numpy(smoothnoise(xv, yv)).type(dtype=torch.cfloat).to(device="cuda")
xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device="cuda")
yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device="cuda")
nR = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
kmax = np.pi / dx
dk = 2 * kmax / N
k = torch.arange(-kmax, kmax, dk, device="cuda").type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing="xy")
kTimeEvo = torch.exp(-0.5j * hbar * (kxv * kxv + kyv * kyv) * dt / m)
pump = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
points = makeSunGrid(80, 4)
for p in points:
    pump += pumpStrength * tgauss(xv - p[0], yv - p[1])

constpart = constV + G * eta * pump / Gamma


@torch.jit.script
def tnormSqr(x):
    return x.conj() * x


@torch.jit.script
def V(psi, nR, constPart, alpha):
    return constPart + alpha * tnormSqr(psi) + (0.002 + 0.5j * 0.016) * nR


@torch.jit.script
def halfRStepPsi(psi, nR, constPart, alpha):
    return psi * torch.exp(-0.5j * 0.05 * V(psi, nR, constPart, alpha))


@torch.jit.script
def halfStepNR(psi, nR, pump):
    return (
        math.exp(-0.5 * 0.05 * 0.1)
        * torch.exp((-0.5 * 0.05 * 0.016) * tnormSqr(psi))
        * nR
        + pump * 0.05 * 0.5
    )


@torch.jit.script
def stepNR(psi, nR, pump):
    return (
        math.exp(-0.05 * 0.1) * torch.exp((-0.05 * 0.016) * tnormSqr(psi)) * nR
        + pump * 0.05
    )


@torch.jit.script
def step(psi0, nR0, kTimeEvo, constPart, pump, alpha):
    psi = halfRStepPsi(psi0, nR0, constPart, alpha)
    psi = tfft.ifft2(kTimeEvo * tfft.fft2(psi0))
    nR = halfStepNR(psi, nR0, pump)
    psi = halfRStepPsi(psi, nR, constPart, alpha)
    nR = halfStepNR(psi, nR, pump)
    return psi, nR


@torch.jit.script
def altstep(psi0, nR0, kTimeEvo, constPart, pump, alpha):
    psi = halfRStepPsi(psi0, nR0, constPart, alpha)
    psi = tfft.ifft2(kTimeEvo * tfft.fft2(psi0))
    psi = halfRStepPsi(psi, nR0, constPart, alpha)
    nR = stepNR(psi, nR0, pump)
    return psi, nR


@torch.jit.script
def runSim(psi, nR, kTimeEvo, constPart, pump, alpha):
    for _ in range(4096):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump, alpha)
    return psi, nR


with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("testing"):
        for _ in range(3):
            psi, nR = runSim(
                psi, nR, kTimeEvo, constpart, pump, torch.tensor([alpha], device="cuda")
            )

            psidata = psi.detach().cpu().numpy()
            imshowBoilerplate(
                npnormSqr(psidata),
                filename="bleh",
                xlabel="x",
                ylabel="y",
                title="ehh",
                extent=[startX, endX, startX, endX],
                aspect="equal",
            )
print(prof.key_averages().table(sort_by="self_cuda_time_total"))


chime.theme("sonic")
chime.success()
