import numpy as np

from src.penrose import goldenRatio

gammalp = 0.2
constV = -0.5j * gammalp
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 25
sigma = 2
dt = 0.05
Gamma = 0.1
eta = 1
dt = 0.05
hbar = 6.582119569e-1  # meV * ps
m = 0.32
N = 1024
startX = -180
endX = 180
dx = (endX - startX) / N
prerun = 12000
nframes = 1024
rhomblength0 = 24
rhomblength1 = 12
rad0 = rhomblength0 * goldenRatio**4
rad1 = rhomblength1 * goldenRatio**4
ndistances = 36
dr = (rad0 - rad1) / ndistances
kmax = np.pi / dx
dk = 2 * kmax / N

a = f"""
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
params = {{
    "gammalp": {gammalp},
    "alpha": {alpha},
    "G": {G},
    "R": {R},
    "Gamma": {Gamma},
    "eta": {eta},
    "m": {m},
    "N": {N},
    "startX": {startX},
    "endX": {endX},
    "prerun": {prerun},
    "nframes": {nframes},
    "rhomblength0": {rhomblength0},
    "rhomblength1": {rhomblength1},
    "ndistances": {ndistances}
}}
with open(os.path.join(basedir, "parameters.json"), "w") as f:
    json.dump(params, f)


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
        npolars[i] = torch.sum(tnormSqr(psi).real)
        spectrum[i] = torch.sum(psi)
    return psi, nR


nR = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
k = torch.arange({-kmax}, {kmax}, {dk}, device='cuda').type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing='xy')
kTimeEvo = torch.exp(-0.5j * {hbar * dt / m} * (kxv * kxv + kyv * kyv))
radii = torch.arange({rad0}, {rad1}, {-dr})
bleh = torch.zeros(({N}, {ndistances}), dtype=torch.float, device='cuda')
for j, r in enumerate(radii):
    x = np.arange({startX}, {endX}, {dx})
    xv, yv = np.meshgrid(x, x)
    dampingscale = {endX * endX * 3}
    damping = 0*np.cosh((xv*xv + yv*yv) / dampingscale) - 1
    imshowBoilerplate(
            damping.real, "dampingpotential", "x", "y", [{startX}, {endX}, {startX}, {endX}]
            )
    damping = torch.from_numpy(damping).type(dtype=torch.cfloat).to(device='cuda')
    psi = torch.from_numpy(smoothnoise(xv, yv)).type(dtype=torch.cfloat).to(device='cuda')
    xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device='cuda')
    yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device='cuda')
    nR = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
    pump = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
    points = makeSunGrid(r, 4)
    for p in points:
        pump += {pumpStrength} * tgauss(xv - p[0], yv - p[1], s=1.2)
    
    constpart = {constV} - 0.5j * damping + {G * eta / Gamma} * pump
    spectrumgpu = torch.zeros(({nframes}), dtype=torch.cfloat, device="cuda")
    npolarsgpu = torch.zeros(({nframes}), dtype=torch.float, device="cuda")
    psi, nR = runSim(psi, nR, kTimeEvo, constpart, pump, npolarsgpu, spectrumgpu)
    spectrumgpu = tfft.fftshift(tfft.ifft(spectrumgpu))
    bleh[:, j] = tnormSqr(spectrumgpu).real / torch.max(tnormSqr(spectrumgpu).real)
    npolars = npolarsgpu.detach().cpu().numpy()
    fig, ax = figBoilerplate()
    ax.plot({dt} * np.arange({nframes}), npolars * {dx * dx})
    name = f"npolarsr{{r}}"
    plt.savefig(os.path.join(basedir, f"{{name}}.pdf"))
    plt.close()
    kpsidata = tnormSqr(tfft.fftshift(tfft.fft2(psi))).real.detach().cpu().numpy()
    rpsidata = tnormSqr(psi).real.detach().cpu().numpy()
    extentr = np.array([{startX}, {endX}, {startX}, {endX}])
    extentk = np.array([{-kmax / 2}, {kmax / 2}, {-kmax / 2}, {kmax / 2}])
    imshowBoilerplate(
        rpsidata,
        filename=os.path.join(basedir, f"rr{{r}}"),
        xlabel="x",
        ylabel="y",
        title="ehh",
        extent=extentr,
        aspect="equal",
    )
    kpsidata = kpsidata[
        {N // 4 - 1} : {N - N // 4},
        {N // 4 - 1} : {N - N // 4},
    ]
    imshowBoilerplate(
        np.log(kpsidata + np.exp(-20)),
        filename=os.path.join(basedir, f"klogr{{r}}"),
        xlabel="$k_x$ (µ$m^{-1}$)",
        ylabel=r"$k_y$ (µ$m^{-1}$)",
        title=r"$\\ln(|\\psi_k|^2 + e^{-20})$",
        extent=[{-kmax / 2}, {kmax / 2}, {-kmax / 2}, {kmax / 2}],
        aspect="equal",
    )

bleh = bleh.detach().cpu().numpy()
np.save("wasistlos", bleh)
ommax = {hbar / dt} * np.pi
imshowBoilerplate(
    bleh[{int(nframes * 0.5)} : {int(0.55 * nframes)}, ::-1],
    filename=os.path.join(basedir, f"intensity{rad0}"),
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$I(E, d)$",
    extent=[{rhomblength1}, {rhomblength0}, 0, ommax / 10],
)

imshowBoilerplate(
    np.log(bleh[{int(nframes * 0.5)} : {int(0.55 * nframes)}, ::-1] + np.exp(-20)),
    filename=os.path.join(basedir, f"intensitylog{rad0}"),
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$\\ln(I(E, d))$",
    extent=[{rhomblength1}, {rhomblength0}, 0, ommax / 10],
)

chime.theme("sonic")
chime.success()
t2 = time.time()
"""

with open(".run.py", "w") as f:
    f.write(a)
