import numpy as np

from src.penrose import goldenRatio

gammalp = 0.2
constV = -0.5j * gammalp
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 14
dt = 0.05
Gamma = 0.1
eta = 2
dt = 0.05
hbar = 6.582119569e-1  # meV * ps
m = 0.32
N = 1024
startX = -150
endX = 150
dx = (endX - startX) / N
prerun = 8000
nframes = 1024
rhomblength0 = 10
rhomblength1 = 18
ndistances = 16
dr = (rhomblength1 - rhomblength0) / ndistances
kmax = np.pi / dx
dk = 2 * kmax / N
sigmax = 2
sigmay = 2

a = f"""
import math
import json
import os
import shutil
import time
from pathlib import Path
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as tfft
from desktop_notifier import DesktopNotifier

from src.solvers import figBoilerplate, npnormSqr, imshowBoilerplate, smoothnoise, tgauss
from src.penrose import filterByRadius, makeSunGrid

t1 = time.time()
now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
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
    "ndistances": {ndistances},
    "sigmax": {sigmax},
    "sigmay": {sigmay},
}}


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

basis = {rhomblength0} * np.array([[np.cos(i * np.pi / 5), np.sin(i * np.pi / 5)] for i in range(10)])

thin = np.array([[0, 0], basis[2], basis[3], basis[2] + basis[3]])

thin -= np.mean(thin, axis=0)

thick = np.array([[0, 0], basis[1], basis[4], basis[4] + basis[1]])

thick -= np.mean(thick, axis=0)

thinthin = np.array(
    [[0, 0], basis[2], basis[3], basis[2] + basis[3], basis[2] + basis[1], basis[1]]
)


thinthin -= np.mean(thinthin, axis=0)

thickthin = np.array(
    [[0, 0], basis[1], basis[4], basis[4] + basis[1], basis[1] + basis[0], basis[0]]
)

thickthin -= np.mean(thickthin, axis=0)

thickthick = np.array(
    [[0, 0], basis[2], basis[5], basis[5] + basis[2], basis[0] + basis[2], basis[0]]
)

thickthick -= np.mean(thickthick, axis=0)

thinthickthin = np.array(
    [
        [0, 0],
        basis[1],
        basis[4],
        basis[1] + basis[4],
        basis[0],
        basis[1] + basis[0],
        basis[5],
        basis[5] + basis[4],
    ]
)

thinthickthin -= np.mean(thinthickthin, axis=0)

penrose0 = makeSunGrid({rhomblength0 * goldenRatio**4}, 4)
penrose1 = filterByRadius(penrose0, 50)
penrose2 = filterByRadius(penrose0, 30)

point = np.array([[10,0]])

setupdict = {{
    "thin": thin,
    "thick": thick,
    "thinthin": thinthin,
    "thickthin": thickthin,
    "thickthick": thickthick,
    "thinthickthin": thinthickthin,
    "penrose0": penrose0,
    "penrose1": penrose1,
    "penrose2": penrose2,
#    "point": point,
}}

nR = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
k = torch.arange({-kmax}, {kmax}, {dk}, device='cuda').type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing='xy')
kTimeEvo = torch.exp(-0.5j * {hbar * dt / m} * (kxv * kxv + kyv * kyv))
rhomblengths = torch.arange({rhomblength0}, {rhomblength1}, {dr})
for key in setupdict:
    bleh = np.zeros(({nframes}, {ndistances}))
    orgpoints = setupdict[key]
    basedir = os.path.join("graphs", key, day, timeofday)
    Path(basedir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(basedir, "parameters.json"), "w") as f:
        json.dump(params, f)
    for j, r in enumerate(rhomblengths):
        x = np.arange({startX}, {endX}, {dx})
        xv, yv = np.meshgrid(x, x)
        # dampingscale = {endX * endX * 3}
        # damping = 0*(np.cosh((xv*xv + yv*yv) / dampingscale) - 1)
        # imshowBoilerplate(
        #         damping.real, "dampingpotential", "x", "y", [{startX}, {endX}, {startX}, {endX}]
        #         )
        # damping = torch.from_numpy(damping).type(dtype=torch.cfloat).to(device='cuda')
        psi = torch.from_numpy(smoothnoise(xv, yv)).type(dtype=torch.cfloat).to(device='cuda')
        xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device='cuda')
        yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device='cuda')
        nR = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
        pump = torch.zeros(({N}, {N}), device='cuda', dtype=torch.cfloat)
        points = (r / {rhomblength0}) * orgpoints
        for p in points:
            pump += {pumpStrength} * tgauss(xv - p[0],
                                            yv - p[1],
                                            sigmax={sigmax},
                                            sigmay={sigmay})

        constpart = {constV} + {G * eta / Gamma} * pump
        spectrumgpu = torch.zeros(({nframes}), dtype=torch.cfloat, device="cuda")
        npolarsgpu = torch.zeros(({nframes}), dtype=torch.float, device="cuda")
        psi, nR = runSim(psi, nR, kTimeEvo, constpart, pump, npolarsgpu, spectrumgpu)
        spectrumgpu = tfft.fftshift(tfft.ifft(spectrumgpu))
        spectrumnp = spectrumgpu.detach().cpu().numpy()
        bleh[:, j] = npnormSqr(spectrumnp) / np.max(npnormSqr(spectrumnp))
        npolars = npolarsgpu.detach().cpu().numpy()
        np.save(os.path.join(basedir, f"npolarsr{{r:.2f}}"), npolars)
        kpsidata = tnormSqr(tfft.fftshift(tfft.fft2(psi))).real.detach().cpu().numpy()
        rpsidata = tnormSqr(psi).real.detach().cpu().numpy()
        extentr = np.array([{startX}, {endX}, {startX}, {endX}])
        extentk = np.array([{-kmax}, {kmax}, {-kmax}, {kmax}])
        np.save(os.path.join(basedir, f"psidata{{r:.2f}}"),
                {{"kpsidata": kpsidata,
                 "rpsidata": rpsidata,
                 "extentr": extentr,
                 "extentk": extentk,
                 }})
    
    np.save(os.path.join(basedir, "spectra"), 
            {{"spectra": bleh,
             "extent": [{rhomblength0},
                        {rhomblength1},
                        -{np.pi * hbar / dt},
                        {np.pi * hbar / dt}]}})

t2 = time.time()
print(f"finished in {{t2 - t1}} seconds")
"""

with open(".run.py", "w") as f:
    f.write(a)
