
import math
import json
import os
import shutil
import time
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
params = {
    "gammalp": 0.2,
    "alpha": 0.0004,
    "G": 0.002,
    "R": 0.016,
    "Gamma": 0.1,
    "eta": 1,
    "m": 0.32,
    "N": 1024,
    "startX": -180,
    "endX": 180,
    "prerun": 12000,
    "nframes": 1024,
    "rhomblength0": 24,
    "rhomblength1": 12,
    "ndistances": 36
}
with open(os.path.join(basedir, "parameters.json"), "w") as f:
    json.dump(params, f)


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
    for _ in range(12000):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
    for i in range(1024):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
        npolars[i] = torch.sum(tnormSqr(psi).real)
        spectrum[i] = torch.sum(psi)
    return psi, nR


nR = torch.zeros((1024, 1024), device='cuda', dtype=torch.cfloat)
k = torch.arange(-8.936085770210967, 8.936085770210967, 0.017453292519943295, device='cuda').type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing='xy')
kTimeEvo = torch.exp(-0.5j * 0.102845618265625 * (kxv * kxv + kyv * kyv))
radii = torch.arange(164.49844718999245, 82.24922359499622, -2.284700655416562)
bleh = torch.zeros((1024, 36), dtype=torch.float, device='cuda')
for j, r in enumerate(radii):
    x = np.arange(-180, 180, 0.3515625)
    xv, yv = np.meshgrid(x, x)
    dampingscale = 97200
    damping = np.cosh((xv*xv + yv*yv) / dampingscale) - 1
    imshowBoilerplate(
            damping.real, "dampingpotential", "x", "y", [-180, 180, -180, 180]
            )
    damping = torch.from_numpy(damping).type(dtype=torch.cfloat).to(device='cuda')
    psi = torch.from_numpy(smoothnoise(xv, yv)).type(dtype=torch.cfloat).to(device='cuda')
    xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device='cuda')
    yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device='cuda')
    nR = torch.zeros((1024, 1024), device='cuda', dtype=torch.cfloat)
    pump = torch.zeros((1024, 1024), device='cuda', dtype=torch.cfloat)
    points = makeSunGrid(r, 4)
    for p in points:
        pump += 24 * tgauss(xv - p[0], yv - p[1], s=1.2)
    
    constpart = -0.1j - 0.5j * damping + 0.02 * pump
    spectrumgpu = torch.zeros((1024), dtype=torch.cfloat, device="cuda")
    npolarsgpu = torch.zeros((1024), dtype=torch.float, device="cuda")
    psi, nR = runSim(psi, nR, kTimeEvo, constpart, pump, npolarsgpu, spectrumgpu)
    spectrumgpu = tfft.fftshift(tfft.ifft(spectrumgpu))
    bleh[:, j] = tnormSqr(spectrumgpu).real / torch.max(tnormSqr(spectrumgpu).real)
    npolars = npolarsgpu.detach().cpu().numpy()
    fig, ax = figBoilerplate()
    ax.plot(0.05 * np.arange(1024), npolars * 0.12359619140625)
    name = f"npolarsr{r}"
    plt.savefig(os.path.join(basedir, f"{name}.pdf"))
    plt.close()
    kpsidata = tnormSqr(tfft.fftshift(tfft.fft2(psi))).real.detach().cpu().numpy()
    rpsidata = tnormSqr(psi).real.detach().cpu().numpy()
    extentr = np.array([-180, 180, -180, 180])
    extentk = np.array([-4.468042885105484, 4.468042885105484, -4.468042885105484, 4.468042885105484])
    imshowBoilerplate(
        rpsidata,
        filename=os.path.join(basedir, f"rr{r}"),
        xlabel="x",
        ylabel="y",
        title="ehh",
        extent=extentr,
        aspect="equal",
    )
    kpsidata = kpsidata[
        255 : 768,
        255 : 768,
    ]
    imshowBoilerplate(
        np.log(kpsidata + np.exp(-20)),
        filename=os.path.join(basedir, f"klogr{r}"),
        xlabel="$k_x$ (µ$m^-1$)",
        ylabel=r"$k_y$ (µ$m^-1$)",
        title=r"$\ln(|\psi_k|^2 + e^-20)$",
        extent=[-4.468042885105484, 4.468042885105484, -4.468042885105484, 4.468042885105484],
        aspect="equal",
    )

bleh = bleh.detach().cpu().numpy()
np.save("wasistlos", bleh)
ommax = 13.164239138 * np.pi
imshowBoilerplate(
    bleh[512 : 563, ::-1],
    filename=os.path.join(basedir, f"intensity164.49844718999245"),
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$I(E, d)$",
    extent=[12, 24, 0, ommax / 10],
)

imshowBoilerplate(
    np.log(bleh[512 : 563, ::-1] + np.exp(-20)),
    filename=os.path.join(basedir, f"intensitylog164.49844718999245"),
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$\ln(I(E, d))$",
    extent=[12, 24, 0, ommax / 10],
)

chime.theme("sonic")
chime.success()
t2 = time.time()
