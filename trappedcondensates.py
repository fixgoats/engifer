import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import torch
import torch.fft as tfft

from src.solvers import hbar, runSim

pumpStrengths = np.arange(15, 24, 7000)
rs = np.linspace(5, 8, 1)
for p in pumpStrengths:
    for r in rs:
        now = gmtime()
        day = strftime("%Y-%m-%d", now)
        timeofday = strftime("%H.%M", now)
        basedir = os.path.join("data/trap", day, timeofday)
        Path(basedir).mkdir(parents=True, exist_ok=True)

        gammalp = 0.2
        constV = -0.5j * gammalp
        alpha = 0.0004
        G = 0.002
        R = 0.016
        pumpStrength = p
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
        nPolarSamples = spectrumSamples + prerun
        hbar = 6.582119569e-1  # meV * ps
        m = 0.32
        startX = -64
        endX = 64
        dx = (endX - startX) / nElementsX

        kmax = np.pi / dx
        dk = 2 * kmax / nElementsX
        beta0 = 1.05
        L0 = 2.2
        r0 = r
        seed = 2001124
        # seed = None

        def pumpprofile(x, y, L, r, beta):
            return (L * L) ** 2 / ((x**2 + beta * y**2 - r**2) ** 2 + L**4)

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
        gen = torch.random.manual_seed(seed=seed)
        psi = 2 * torch.rand(
            (nElementsX, nElementsX), generator=gen, dtype=torch.cfloat
        ).to(dev10ice="cuda") - (1 + 1j)
        pump1 = pumpStrength * pumpprofile(xv - 15, yv, L0, r0, beta0)
        pump2 = pumpStrength * pumpprofile(xv + 15, yv, L0, r0, beta0)
        pump = pump1 + pump2
        constpart = constV + (G * eta / Gamma) * pump

        nPolars = torch.zeros((nPolarSamples), dtype=torch.float, device="cuda")
        spectrum = torch.zeros((spectrumSamples), dtype=torch.cfloat, device="cuda")
        nExcitons = torch.zeros((nPolarSamples), dtype=torch.float, device="cuda")

        print("Setup done, running simulation")
        psi, nR = runSim(
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
            nPolars,
            nExcitons,
            spectrum,
            prerun,
            spectrumSamples,
            nPolarSamples,
            sampleSpacing,
        )

        rpsidata = psi.detach().cpu().numpy()
        kpsidata = tfft.fftshift(tfft.fft2(psi)).detach().cpu().numpy()
        spectrum = spectrum.detach().cpu().numpy()
        nExcitons = nExcitons.detach().cpu().numpy()
        nPolars = nPolars.detach().cpu().numpy()
        extentr = np.array([startX, endX, startX, endX])
        extentk = np.array([-kmax, kmax, -kmax, kmax])
        np.savez(
            os.path.join(basedir, f"psidatar{r}p{p}"),
            rpsidata=rpsidata,
            kpsidata=kpsidata,
            extentr=extentr,
            extentk=extentk,
            spectrum=spectrum,
            nExcitons=nExcitons,
            nPolars=nPolars,
        )

        t2 = time.time()
        print(f"Finished in {t2 - t1} seconds")
