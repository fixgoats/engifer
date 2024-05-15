
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
from src.penrose import filterByRadius, makeSunGrid

t1 = time.time()
now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
params = {
    "gammalp": 0.2,
    "alpha": 0.0004,
    "G": 0.002,
    "R": 0.016,
    "Gamma": 0.1,
    "eta": 2,
    "m": 0.32,
    "N": 1024,
    "startX": -150,
    "endX": 150,
    "prerun": 8000,
    "nframes": 1024,
    "rhomblength0": 10,
    "rhomblength1": 18,
    "ndistances": 16,
    "sigmax": 2,
    "sigmay": 2,
}


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
    for _ in range(8000):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
    for i in range(1024):
        psi, nR = step(psi, nR, kTimeEvo, constPart, pump)
        npolars[i] = torch.sum(tnormSqr(psi).real)
        spectrum[i] = torch.sum(psi)
    return psi, nR

basis = 10 * np.array([[np.cos(i * np.pi / 5), np.sin(i * np.pi / 5)] for i in range(10)])

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

penrose0 = makeSunGrid(68.54101966249685, 4)
penrose1 = filterByRadius(penrose0, 50)
penrose2 = filterByRadius(penrose0, 30)

point = np.array([[10,0]])

setupdict = {
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
}

nR = torch.zeros((1024, 1024), device='cuda', dtype=torch.cfloat)
k = torch.arange(-10.72330292425316, 10.72330292425316, 0.020943951023931952, device='cuda').type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing='xy')
kTimeEvo = torch.exp(-0.5j * 0.102845618265625 * (kxv * kxv + kyv * kyv))
rhomblengths = torch.arange(10, 18, 0.5)
for key in setupdict:
    bleh = np.zeros((1024, 16))
    orgpoints = setupdict[key]
    basedir = os.path.join("graphs", key, day, timeofday)
    Path(basedir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(basedir, "parameters.json"), "w") as f:
        json.dump(params, f)
    for j, r in enumerate(rhomblengths):
        x = np.arange(-150, 150, 0.29296875)
        xv, yv = np.meshgrid(x, x)
        # dampingscale = 67500
        # damping = 0*(np.cosh((xv*xv + yv*yv) / dampingscale) - 1)
        # imshowBoilerplate(
        #         damping.real, "dampingpotential", "x", "y", [-150, 150, -150, 150]
        #         )
        # damping = torch.from_numpy(damping).type(dtype=torch.cfloat).to(device='cuda')
        psi = torch.from_numpy(smoothnoise(xv, yv)).type(dtype=torch.cfloat).to(device='cuda')
        xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device='cuda')
        yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device='cuda')
        nR = torch.zeros((1024, 1024), device='cuda', dtype=torch.cfloat)
        pump = torch.zeros((1024, 1024), device='cuda', dtype=torch.cfloat)
        points = (r / 10) * orgpoints
        for p in points:
            pump += 14 * tgauss(xv - p[0],
                                            yv - p[1],
                                            sigmax=2,
                                            sigmay=2)

        constpart = -0.1j + 0.04 * pump
        spectrumgpu = torch.zeros((1024), dtype=torch.cfloat, device="cuda")
        npolarsgpu = torch.zeros((1024), dtype=torch.float, device="cuda")
        psi, nR = runSim(psi, nR, kTimeEvo, constpart, pump, npolarsgpu, spectrumgpu)
        spectrumgpu = tfft.fftshift(tfft.ifft(spectrumgpu))
        spectrumnp = spectrumgpu.detach().cpu().numpy()
        bleh[:, j] = npnormSqr(spectrumnp) / np.max(npnormSqr(spectrumnp))
        npolars = npolarsgpu.detach().cpu().numpy()
        np.save(os.path.join(basedir, f"npolarsr{r:.2f}"), npolars)
        kpsidata = tnormSqr(tfft.fftshift(tfft.fft2(psi))).real.detach().cpu().numpy()
        rpsidata = tnormSqr(psi).real.detach().cpu().numpy()
        extentr = np.array([-150, 150, -150, 150])
        extentk = np.array([-10.72330292425316, 10.72330292425316, -10.72330292425316, 10.72330292425316])
        np.save(os.path.join(basedir, f"psidata{r:.2f}"),
                {"kpsidata": kpsidata,
                 "rpsidata": rpsidata,
                 "extentr": extentr,
                 "extentk": extentk,
                 })
    
    np.save(os.path.join(basedir, "spectra"), 
            {"spectra": bleh,
             "extent": [10,
                        18,
                        -41.35667696604003,
                        41.35667696604003]})

chime.theme("sonic")
chime.success()
t2 = time.time()
print(f"finished in {t2 - t1} seconds")
