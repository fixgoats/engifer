import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import torch
import torch.fft as tfft

from src.solvers import runSim, smoothnoise, tgauss

now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
basedir = os.path.join("graphs", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)

seed = None

gammalp = 0.2
constV = -0.5j * gammalp
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 6
dt = 0.05
Gamma = 0.1
eta = 2
dt = 0.05
hbar = 6.582119569e-1  # meV * ps
m = 0.32  # meV ps^2 µm^-2
N = 1024
startX = -50
endX = 50
dx = (endX - startX) / N
prerun = 10000
ntimes = 1024
nframes = 24 * 10
recordInterval = (prerun + ntimes) // nframes
kmax = np.pi / dx
dk = 2 * kmax / N
sigmax = 1.8
sigmay = 1.8

t1 = time.time()
nR = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
k = torch.arange(-kmax, kmax, dk, device="cuda").type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing="xy")
kTimeEvo = torch.exp(-0.5j * hbar * (dt / m) * (kxv * kxv + kyv * kyv))
x = np.arange(startX, endX, dx)
xv, yv = np.meshgrid(x, x)
psi = torch.from_numpy(smoothnoise(xv, yv)).type(dtype=torch.cfloat).to(device="cuda")
xv = torch.from_numpy(xv).type(dtype=torch.cfloat).to(device="cuda")
yv = torch.from_numpy(yv).type(dtype=torch.cfloat).to(device="cuda")
pump = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
for i in range(4):
    for j in range(4):
        pump += pumpStrength * tgauss(
            xv + 19.8 - 13.2 * j,
            yv + 19.8 - 13.2 * i,
            sigmax,
            sigmay,
        )
constpart = constV + (G * eta / Gamma) * pump
spectrumgpu = torch.zeros((ntimes), dtype=torch.cfloat, device="cuda")
npolarsgpu = torch.zeros(
    ((prerun + ntimes) // 20 + 1), dtype=torch.float, device="cuda"
)
# animationFeed = torch.zeros((N, N, nframes), dtype=torch.float, device="cuda")
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
    # animationFeed,
)
npolars = npolarsgpu.detach().cpu().numpy()
spectrum = tfft.ifft(spectrumgpu).detach().cpu().numpy()
np.save(os.path.join(basedir, "spectrum"), spectrum)
np.save(os.path.join(basedir, "npolars"), npolars)
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
# animationFeedNp = animationFeed.detach().cpu().numpy()
# np.save(os.path.join(basedir, "animationFeed"), animationFeedNp)
t2 = time.time()
print(f"Finished in {t2 - t1} seconds")
