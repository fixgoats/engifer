import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import torch
import torch.fft as tfft

from src.solvers import runSimPlain

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
pumpStrength = 18
Gamma = 0.1
eta = 2
dt = 0.05
hbar = 6.582119569e-1  # meV * ps
m = 0.32
N = 512
startX = -32
endX = 32
dx = (endX - startX) / N
prerun = 20000
ntimes = 1024
nframes = 24 * 10
recordInterval = (prerun + ntimes) // nframes
kmax = np.pi / dx
dk = 2 * kmax / N
beta0 = 1.05
L0 = 2.6
r0 = 13.2
seed = 2001124


def pumpprofile(x, y, L, r, beta):
    return (L * L) ** 2 / ((x**2 + beta * y**2 - r**2) ** 2 + L**4)


t1 = time.time()
nR = torch.zeros((N, N), device="cuda", dtype=torch.cfloat)
k = torch.arange(-kmax, kmax, dk, device="cuda").type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing="xy")
kTimeEvo = torch.exp(-0.5j * hbar * (dt / m) * (kxv * kxv + kyv * kyv))
x = torch.arange(startX, endX, dx)
xv, yv = torch.meshgrid(x, x, indexing="xy")
xv = xv.type(dtype=torch.cfloat).to(device="cuda")
yv = yv.type(dtype=torch.cfloat).to(device="cuda")
gen = torch.random.manual_seed(seed=seed)
psi = 2 * torch.rand((N, N), generator=gen, dtype=torch.cfloat).to(device="cuda") - (
    1 + 1j
)
pump = pumpStrength * pumpprofile(xv, yv, L0, r0, beta0)
constpart = constV + (G * eta / Gamma) * pump

print("Setup done, running simulation")
psi, nR = runSimPlain(
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
    prerun,
)

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
