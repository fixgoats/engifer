from src.penrose import goldenRatio

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
N = 1024
startX = -100
endX = 100
prerun = 4096
nframes = 1024
rhomblength0 = 24
rhomblength1 = 12
rad0 = rhomblength0 * goldenRatio**3
rad1 = rhomblength1 * goldenRatio**3
ndistances = 36
dr = (rad0 - rad1) / ndistances

a = f"""
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
    return constPart + {alpha} * tnormSqr(psi) + ({G} + 0.5j * {R}) * nR


@torch.jit.script
def halfRStepPsi(psi, nR, constPart):
    return psi * torch.exp(-0.5j * {dt} * V(psi, nR, constPart))


@torch.jit.script
def halfStepNR(psi, nR, pump):
    return (
        math.exp(-0.5 * {dt} * {Gamma})
        * torch.exp((-0.5 * {dt} * {R}) * tnormSqr(psi))
        * nR
        + pump * {dt} * 0.5
    )


@torch.jit.script
def stepNR(psi, nR, pump):
    return (
        math.exp(-{dt} * {Gamma}) * torch.exp((-{dt} * {R}) * tnormSqr(psi)) * nR
        + pump * {dt}
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
    for _ in range({prerun}):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
    for i in range({nframes}):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
        npolars[i] = torch.sum(tnormSqr(psi))
        spectrum[i] = torch.sum(psi)
    return psi, nR


dx = ({endX} - {startX}) / {N}
x = np.arange({startX}, {endX}, dx)
xv, yv = np.meshgrid(x, x)

xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device='cuda')
yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device='cuda')
nR = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
kmax = np.pi / dx
dk = 2 * kmax / {N}
k = torch.arange(-kmax, kmax, dk, device='cuda').type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing='xy')
kTimeEvo = torch.exp(-0.5j * {hbar * dt / m} * (kxv * kxv + kyv * kyv))
radii = torch.arange({rad0}, {rad1}, {-dr})
print(radii)
bleh = torch.zeros(({N}, {N}), dtype=torch.cfloat, device='cuda')
for j, r in enumerate(radii):
    psi = torch.from_numpy(smoothnoise(xv, yv)).type(dtype=torch.cfloat).to(device='cuda')
    nR = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
    pump = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
    points = makeSunGrid(r, 4)
    for p in points:
        pump += {pumpStrength} * tgauss(xv - p[0], yv - p[1])
    
    constpart = {constV} + {G * eta / Gamma} * pump
    spectrumgpu = torch.zeros(({nframes}), dtype=torch.cfloat, device="cuda")
    npolarsgpu = torch.zeros(({nframes}), dtype=torch.cfloat, device="cuda")
    psi, nR = runSim(psi, nR, kTimeEvo, constpart, pump, npolarsgpu, spectrumgpu)
    bleh[:, j] = tnormSqr(spectrumgpu) / torch.max(tnormSqr(spectrumgpu).real)
    npolars = npolarsgpu.detach().cpu().numpy()
    fig, ax = figBoilerplate()
    ax.plot({dt} * np.arange({nframes}), npolars * dx * dx)
    name = f"npolarsr{{r}}"
    plt.savefig(os.path.join(basedir, f"{{name}}.pdf"))
    plt.close()
    kpsidata = tnormSqr(tfft.fftshift(tfft.fft2(psi))).detach().cpu().numpy()
    rpsidata = tnormSqr(psi).detach().cpu().numpy()
    extentr = np.array([{startX}, {endX}, {startX}, {endX}])
    extentk = np.array([-kmax / 2, kmax / 2, -kmax / 2, kmax / 2])
    imshowBoilerplate(
        rpsidata,
        filename=os.path.join(basedir, f"rr{{r}}.pdf"),
        xlabel='x',
        ylabel='y',
        title='ehh',
        extent=extentr,
        aspect='equal',
    )
    kpsidata = kpsidata[
        {N // 4 - 1} : {N - N // 4},
        {N // 4 - 1} : {N - N // 4},
    ]
    imshowBoilerplate(
        np.log(kpsidata + np.exp(-20)),
        filename=os.path.join(basedir, f"klogr{{r}}.pdf"),
        xlabel="$k_x$ (µ$m^{-1}$)",
        ylabel=r"$k_y$ (µ$m^{-1}$)",
        title=r"$\\ln(|\\psi_k|^2 + e^{-20})$",
        extent=[-kmax / 2, kmax / 2, -kmax / 2, kmax / 2],
        aspect="equal",
    )

ommax = {hbar / dt} * np.pi
imshowBoilerplate(
    bleh[{int(nframes * 0.5)} : {int(0.55 * nframes)}, ::-1],
    filename=os.path.join(basedir, f"intensity{rad0}.pdf"),
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$I(E, d)$",
    extent=[{rhomblength1}, {rhomblength0}, 0, ommax / 10],
)

imshowBoilerplate(
    np.log(bleh[{int(nframes * 0.5)} : {int(0.55 * nframes)}, ::-1]),
    filename=os.path.join(basedir, f"intensitylog{rad0}.pdf"),
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$\\ln(I(E, d))$",
    extent=[{rhomblength1}, {rhomblength0}, 0, ommax / 10],
)

chime.theme("sonic")
chime.success()
"""

with open(".run.py", "w") as f:
    f.write(a)
