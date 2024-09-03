import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import torch
import torch.fft as tfft

from src.penrose import goldenRatio
from src.solvers import runSim, smoothnoise, tgauss

now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
basedir = os.path.join("data", "square", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)

seed = 20011204

gammalp = 0.2
constV = -0.5j * gammalp
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 8.5
dt = 0.1
Gamma = 0.1
eta = 1
dt = 0.05
hbar = 6.582119569e-1  # meV * ps
m = 0.32  # meV ps^2 µm^-2
N = 1024
D = 14
divs = 6
radius = D * goldenRatio**divs
startX = -(radius + 30)
endX = radius + 30
dx = (endX - startX) / N
prerun = 30000 // 20
ntimes = 1024
nframes = 24 * 15
recordInterval = (prerun + ntimes) // nframes
kmax = np.pi / dx
dk = 2 * kmax / N
sigmax = 1.27
sigmay = 1.27

t1 = time.time()
nR = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
k = torch.arange(-kmax, kmax, dk, device="cuda").type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing="xy")
kTimeEvo = torch.exp(-0.5j * hbar * (dt / m) * (kxv * kxv + kyv * kyv))
x = np.arange(startX, endX, dx)
xv, yv = np.meshgrid(x, x)

rng = np.random.default_rng(seed)
psi = (
    torch.from_numpy(smoothnoise(xv, yv, rng))
    .type(dtype=torch.cfloat)
    .to(device="cuda")
)
xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device="cuda")
yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device="cuda")

startingPoint = ((endX + startX) - D * 35) / 2
pump = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
orgpoints = np.array(
    [
        ([startingPoint + D * i, startingPoint + D * j])
        for i in range(35)
        for j in range(35)
    ]
)

pump = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
for p in orgpoints:
    pump += pumpStrength * tgauss(
        xv - p[0],
        yv - p[1],
        sigmax,
        sigmay,
    )

constpart = constV + (G * eta / Gamma) * pump
spectrumgpu = torch.zeros((ntimes), dtype=torch.cfloat, device="cuda")
npolarsgpu = torch.zeros(
    (prerun + (ntimes // 10) + 1), dtype=torch.float, device="cuda"
)
# animateTensor = torch.zeros((N // 2, N // 2, nframes), dtype=torch.float, device="cuda")
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
    npolarsgpu,
    spectrumgpu,
    prerun,
    ntimes,
    # recordInterval,
    # animateTensor,
)
npolars = npolarsgpu.detach().cpu().numpy()
spectrum = tfft.ifft(spectrumgpu).detach().cpu().numpy()
np.save(os.path.join(basedir, "spectrum"), spectrum)
np.save(os.path.join(basedir, "npolars"), npolars)
# animatearray = animateTensor.detach().cpu().numpy()
# np.save(os.path.join(basedir, "animatearray"), animatearray)
rpsidata = psi.detach().cpu().numpy()
kpsidata = tfft.fftshift(tfft.fft2(psi)).detach().cpu().numpy()
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
print(f"Finished in {t2 - t1} seconds")
