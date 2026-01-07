import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import torch
import torch.fft as tfft

from src.solvers import hbar, stepWithRK4

pumpStrengths = np.linspace(9, 14, 3)
rs = np.linspace(4.5, 5.5, 2)
now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
basedir = os.path.join("data/trap", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)


def pumpprofile(x, y, L, r, beta):
    return (L * L) ** 2 / ((x**2 + beta * y**2 - r**2) ** 2 + (L * L) ** 2)


# seed = 2001124
for d in [12.2]:
    seed = None
    if seed is not None:
        gen = torch.random.manual_seed(seed=seed)
    else:
        gen = torch.Generator()
    gammalp = 0.2
    constV = -0.5j * gammalp
    alpha = 0.0004
    G = 0.002
    R = 0.016
    pumpStrength = 9.2
    Gamma = 0.1
    eta = 2
    spectrumSamples = 512
    nElementsX = 512
    Emax = 3
    dE = 2 * Emax / spectrumSamples
    dom = dE / hbar
    ommax = Emax / hbar
    T = 2 * np.pi / dom
    dt = 0.05
    sampleSpacing = int((T / spectrumSamples) / dt)
    prerun = 10000 // sampleSpacing
    print(prerun)
    nPolarSamples = spectrumSamples + prerun
    print(nPolarSamples * sampleSpacing)
    hbar = 6.582119569e-1  # meV * ps
    m = 0.32
    startX = -64
    endX = 64
    dx = (endX - startX) / nElementsX

    kmax = np.pi / dx
    dk = 2 * kmax / nElementsX
    beta0 = 1.0
    L0 = 2.2
    r0 = 5.1

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
    psi = 0.2 * torch.rand(
        (nElementsX, nElementsX), generator=gen, dtype=torch.cfloat
    ).to(device="cuda") - (0.1 + 0.1j)
    pump1 = pumpStrength * pumpprofile(xv - d, yv, L0, r0, beta0)
    pump2 = pumpStrength * pumpprofile(xv + d, yv, L0, r0, beta0)
    pump = (pump1 + pump2).real
    constpart = constV + (G * eta / Gamma) * pump

    nPolars = torch.zeros((nPolarSamples), dtype=torch.float, device="cuda")
    spectrum = torch.zeros((spectrumSamples), dtype=torch.cfloat, device="cuda")
    nExcitons = torch.zeros((nPolarSamples), dtype=torch.float, device="cuda")

    print("Setup done, running simulation")
    stop = False
    while not stop:
        x = input()
        if x == "q":
            stop == true
        else:
            print(
                f"psisqmax: {torch.max(psi.real*psi.real + psi.imag*psi.imag)}, nRmax: {torch.max(nR)}"
            )
            psi, nR = stepWithRK4(
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
            )
