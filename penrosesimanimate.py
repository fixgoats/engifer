import json
import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import torch
import torch.fft as tfft

from src.penrose import goldenRatio
from src.solvers import runSim, smoothnoise, tgauss

gammalp = 0.2
constV = -0.5j * gammalp
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 8.5
dt = 0.1
Gamma = 0.1
eta = 1
dt = 0.1
hbar = 6.582119569e-1  # meV * ps
m = 0.32
prerun = 30000 // 20
ntimes = 1024
cutoff = 76
D = 14  # rhombi sidelength in microns
divs = 6  # number of times to perform division algorithm
radius = D * goldenRatio**divs
# points = makeSunGrid(radius, divs)
points = np.load("penrosepoints.npy") * (radius / 100)
startX = -(radius + 30)  # give the polaritons some space to decay at the edges
endX = radius + 30
N = 1024  # even at such a large scale a 1024x1024 grid still captures a good amount of detail
dx = (endX - startX) / N
kmax = np.pi / dx
dk = 2 * kmax / N
sigmax = 1.27
sigmay = 1.27
seed = 20011204
nframes = 24 * 15
recordInterval = (prerun + ntimes) // nframes

t1 = time.time()
now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
params = {
    "gammalp": gammalp,
    "alpha": alpha,
    "G": G,
    "R": R,
    "Gamma": Gamma,
    "eta": eta,
    "D": radius,
    "cutoff": cutoff,
    "m": m,
    "N": N,
    "pumpStrength": pumpStrength,
    "startX": startX,
    "endX": endX,
    "sigmax": sigmax,
    "sigmay": sigmay,
}

nR = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
k = torch.arange(-kmax, kmax, dk, device="cuda").type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing="xy")
kTimeEvo = torch.exp(-0.5j * hbar * dt / m * (kxv * kxv + kyv * kyv))
basedir = os.path.join("data", "p6", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)
x = np.arange(startX, endX, dx)
xv, yv = np.meshgrid(x, x)
xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device="cuda")
yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device="cuda")
rng = np.random.default_rng(seed)
psi = (
    torch.from_numpy(smoothnoise(xv, yv, rng))
    .type(dtype=torch.cfloat)
    .to(device="cuda")
)

with open(os.path.join(basedir, "parameters.json"), "w") as f:
    json.dump(params, f)
nR = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
pump = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
for p in points:
    pump += pumpStrength * tgauss(xv - p[0], yv - p[1], sigmax=sigmax, sigmay=sigmay)

constpart = constV + (G * eta / Gamma) * pump
npolarsgpu = torch.zeros(
    (prerun + (ntimes // 10) + 1), dtype=torch.float, device="cuda"
)
spectrumgpu = torch.zeros((ntimes), dtype=torch.cfloat, device="cuda")
# animatetensor = torch.zeros((N // 2, N // 2, nframes), dtype=torch.float, device="cuda")
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
    npolarsgpu,
    spectrumgpu,
    prerun,
    ntimes,
    # recordInterval,
    # animatetensor,
)
npolars = npolarsgpu.detach().cpu().numpy()
np.save(os.path.join(basedir, "npolars"), npolars)
spectrum = tfft.ifft(spectrumgpu).detach().cpu().numpy()
np.save(os.path.join(basedir, "spectrum"), spectrum)
# animatearray = animatetensor.detach().cpu().numpy()
# np.save(os.path.join(basedir, "animatearray"), animatearray)
kpsidata = tfft.fftshift(tfft.fft2(psi)).detach().cpu().numpy()
rpsidata = psi.detach().cpu().numpy()
extentr = np.array([startX, endX, startX, endX])
extentk = np.array([-kmax, kmax, -kmax, kmax])
np.savez(
    os.path.join(basedir, "psidata"),
    rpsidata=rpsidata,
    kpsidata=kpsidata,
    extentr=extentr,
    extentk=extentk,
)

t2 = time.time()
print(f"finished in {t2 - t1} seconds")
