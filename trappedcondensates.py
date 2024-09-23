import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import torch
import torch.fft as tfft

from src.solvers import hbar, newRunSimAnimate

pumpStrengths = np.linspace(9, 14, 3)
rs = np.linspace(4.5, 5.5, 2)
now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
basedir = os.path.join("data/trap", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)


def pumpprofile(x, y, L, r, beta):
    return (L * L) ** 2 / ((x**2 + beta * y**2 - r**2) ** 2 + (L * L) ** 2)


gammalp = 0.2
constV = -0.5j * gammalp
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 9.0
Gamma = 0.1
eta = 2
nFrames = 512
nElementsX = 512
dt = 0.1
frameSpacing = 30
prerun = 5000
hbar = 6.582119569e-1  # meV * ps
m = 0.32
startX = -64
endX = 64
dx = (endX - startX) / nElementsX

kmax = np.pi / dx
dk = 2 * kmax / nElementsX
beta0 = 0.999
L0 = 2.2
r0 = 5.1
# seed = 2001124
extentr = np.array([startX, endX, startX, endX])
extentk = np.array([-kmax, kmax, -kmax, kmax])
for d in [14.05, 14.06, 14.07, 14.08, 14.09, 14.11, 14.12, 14.13]:
    seed = None
    if seed is not None:
        gen = torch.random.manual_seed(seed=seed)
    else:
        gen = torch.Generator()

    t1 = time.time()
    nR = torch.zeros((nElementsX, nElementsX), device="cuda", dtype=torch.float)
    k = torch.arange(-kmax, kmax, dk, device="cuda").type(dtype=torch.cfloat)
    k = tfft.fftshift(k)
    kxv, kyv = torch.meshgrid(k, k, indexing="xy")
    kTimeEvo = torch.exp(-0.5j * hbar * (dt / m) * (kxv * kxv + kyv * kyv))
    x = torch.arange(startX, endX, dx)
    xv, yv = torch.meshgrid(x, x, indexing="xy")
    xv = xv.type(dtype=torch.cfloat).to(device="cuda")
    yv = yv.type(dtype=torch.cfloat).to(device="cuda")
    psi = 2 * torch.rand(
        (nElementsX, nElementsX), generator=gen, dtype=torch.cfloat
    ).to(device="cuda") - (1 + 1j)
    pump1 = pumpStrength * pumpprofile(xv - d, yv, L0, r0, beta0)
    pump2 = pumpStrength * pumpprofile(xv + d, yv, L0, r0, beta0)
    pump = (pump1 + pump2).real
    constpart = constV + (G * eta / Gamma) * pump

    excitonFeed = torch.zeros(
        (nElementsX, nElementsX, nFrames), dtype=torch.float, device="cuda"
    )
    psiFeed = torch.zeros(
        (nElementsX, nElementsX, nFrames), dtype=torch.cfloat, device="cuda"
    )

    print("Setup done, running simulation")
    psi, nR = newRunSimAnimate(
        psi,
        nR,
        kTimeEvo,
        constpart,
        pump,
        dt,
        alpha,
        G,
        R,
        Gamma,
        prerun,
        nFrames,
        frameSpacing,
        psiFeed,
        excitonFeed,
    )

    psiFeed = psiFeed.detach().cpu().numpy()
    excitonFeed = excitonFeed.detach().cpu().numpy()
    kpsidata = tfft.fftshift(tfft.fft2(psi)).detach().cpu().numpy()
    np.savez(
        os.path.join(basedir, f"simdata{d}"),
        psiFeed=psiFeed,
        excitonFeed=excitonFeed,
        extentr=extentr,
        extentk=extentk,
    )

    t2 = time.time()
    print(f"Finished in {t2 - t1} seconds")
