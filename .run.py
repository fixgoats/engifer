
import math
import os
import shutil
from pathlib import Path
from time import gmtime, strftime

import chime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as tfft

from src.solvers import figBoilerplate, npnormSqr, imshowBoilerplate, smoothnoise, tgauss
from src.penrose import makeSunGrid


now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
basedir = os.path.join("graphs", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)

@torch.jit.script
def tnormSqr(x):
    return x.conj() * x


@torch.jit.script
def V(psi, nR, constPart):
    return constPart + 0.0004 * tnormSqr(psi) + (0.002 + 0.5j * 0.016) * nR


@torch.jit.script
def halfRStepPsi(psi, nR, constPart):
    return psi * torch.exp(-0.5j * 0.05 * V(psi, nR, constPart))


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
def step(psi0, nR0, kTimeEvo, constPart, pump):
    psi = halfRStepPsi(psi0, nR0, constPart)
    psi = tfft.ifft2(kTimeEvo * tfft.fft2(psi0))
    nR = halfStepNR(psi, nR0, pump)
    psi = halfRStepPsi(psi, nR, constPart)
    nR = halfStepNR(psi, nR, pump)
    return psi, nR


@torch.jit.script
def altstep(psi0, nR0, kTimeEvo, constPart, pump):
    psi = halfRStepPsi(psi0, nR0, constPart)
    psi = tfft.ifft2(kTimeEvo * tfft.fft2(psi0))
    psi = halfRStepPsi(psi, nR0, constPart)
    nR = stepNR(psi, nR0, pump)
    return psi, nR


@torch.jit.script
def runSim(psi, nR, kTimeEvo, constPart, pump, npolars, spectrum):
    for _ in range(4096):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
    for i in range(1024):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
        npolars[i] = torch.sum(tnormSqr(psi))
        spectrum[i] = torch.sum(psi)
    return psi, nR


dx = (100 - -100) / 1024
x = np.arange(-100, 100, dx)
xv, yv = np.meshgrid(x, x)

xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device='cuda')
yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device='cuda')
nR = torch.zeros((1024, 1024), device='cuda', dtype=torch.cfloat)
kmax = np.pi / dx
dk = 2 * kmax / 1024
k = torch.arange(-kmax, kmax, dk, device='cuda').type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing='xy')
kTimeEvo = torch.exp(-0.5j * 0.102845618265625 * (kxv * kxv + kyv * kyv))
radii = torch.arange(101.66563145999496, 50.83281572999748, -1.4120226591665965)
print(radii)
bleh = torch.zeros((1024, 1024), dtype=torch.cfloat, device='cuda')
for j, r in enumerate(radii):
    psi = torch.from_numpy(smoothnoise(xv, yv)).type(dtype=torch.cfloat).to(device='cuda')
    nR = torch.zeros((1024, 1024), device='cuda', dtype=torch.cfloat)
    pump = torch.zeros((1024, 1024), device='cuda', dtype=torch.cfloat)
    points = makeSunGrid(r, 4)
    for p in points:
        pump += 24 * tgauss(xv - p[0], yv - p[1])
    
    constpart = -0.1j + 0.0 * pump
    spectrumgpu = torch.zeros((1024), dtype=torch.cfloat, device="cuda")
    npolarsgpu = torch.zeros((1024), dtype=torch.cfloat, device="cuda")
    psi, nR = runSim(psi, nR, kTimeEvo, constpart, pump, npolarsgpu, spectrumgpu)
    bleh[:, j] = tnormSqr(spectrumgpu) / torch.max(tnormSqr(spectrumgpu).real)
    npolars = npolarsgpu.detach().cpu().numpy()
    fig, ax = figBoilerplate()
    ax.plot(0.05 * np.arange(1024), npolars * dx * dx)
    name = f"npolarsr{r}"
    plt.savefig(os.path.join(basedir, f"{name}.pdf"))
    plt.close()
    kpsidata = tnormSqr(tfft.fftshift(tfft.fft2(psi))).detach().cpu().numpy()
    rpsidata = tnormSqr(psi).detach().cpu().numpy()
    extentr = np.array([-100, 100, -100, 100])
    extentk = np.array([-kmax / 2, kmax / 2, -kmax / 2, kmax / 2])
    imshowBoilerplate(
        rpsidata,
        filename=os.path.join(basedir, f"rr{r}.pdf"),
        xlabel='x',
        ylabel='y',
        title='ehh',
        extent=extentr,
        aspect='equal',
    )
    kpsidata = kpsidata[
        255 : 768,
        255 : 768,
    ]
    imshowBoilerplate(
        np.log(kpsidata + np.exp(-20)),
        filename=os.path.join(basedir, f"klogr{r}.pdf"),
        xlabel="$k_x$ (µ$m^-1$)",
        ylabel=r"$k_y$ (µ$m^-1$)",
        title=r"$\ln(|\psi_k|^2 + e^-20)$",
        extent=[-kmax / 2, kmax / 2, -kmax / 2, kmax / 2],
        aspect="equal",
    )

ommax = 13.164239138 * np.pi
imshowBoilerplate(
    bleh[512 : 563, ::-1],
    filename=os.path.join(basedir, f"intensity101.66563145999496.pdf"),
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$I(E, d)$",
    extent=[12, 24, 0, ommax / 10],
)

imshowBoilerplate(
    np.log(bleh[512 : 563, ::-1]),
    filename=os.path.join(basedir, f"intensitylog101.66563145999496.pdf"),
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$\ln(I(E, d))$",
    extent=[12, 24, 0, ommax / 10],
)

chime.theme("sonic")
chime.success()
