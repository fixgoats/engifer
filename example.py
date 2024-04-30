import argparse
import os
import tomllib
from datetime import date
from pathlib import Path

import chime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.fft import fft, fftshift, ifft

from src.penrose import goldenRatio, makeSunGrid
from src.solvers import SsfmGPGPU, gauss, hbar, imshowBoilerplate, npnormSqr, tnormSqr

datestamp = date.today()
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--use-cached", action="store_true")
args = parser.parse_args()
if args.config is None and args.use_cached is None:
    exit("Need to specify config")

with open(f"{args.config}", "rb") as f:
    pars = tomllib.load(f)


dev = torch.device("cuda")  # Change this to cpu if you don't have an nvidia gpu
endX = pars["endX"]
startX = pars["startX"]
samplesX = pars["samplesX"]
endY = pars["endY"]
startY = pars["startY"]
samplesY = pars["samplesY"]
dx = (endX - startX) / samplesX
dy = (endY - startY) / samplesY
x = torch.arange(startX, endX, dx)
y = torch.arange(startY, endY, dy)
x = x.type(dtype=torch.cfloat)
y = y.type(dtype=torch.cfloat)
gridX, gridY = torch.meshgrid(x, y, indexing="xy")
kxmax = np.pi / dx
kymax = np.pi / dy
dkx = 2 * kxmax / samplesX
psi = 0.1 * torch.rand((samplesY, samplesX), dtype=torch.cfloat)

constV = -0.5j * pars["gammalp"] * torch.ones((samplesY, samplesX))

pump = pars["pumpStrength"] * (gauss(gridX, gridY, pars["sigma"], pars["sigma"]))
nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
gpsim = SsfmGPGPU(
    dev=dev,
    psi0=psi,
    gridX=gridX,
    gridY=gridY,
    m=pars["m"],
    nR0=nR,
    alpha=pars["alpha"],
    Gamma=pars["Gamma"],
    gammalp=pars["gammalp"],
    R=pars["R"],
    pump=pump,
    G=pars["G"],
    eta=pars["eta"],
    constV=constV,
    dt=pars["dt"],
)  # the constructor takes care of sending all arrays to the appropriate device

nframes = 1024
dispersion = torch.zeros(
    (nframes, samplesX), dtype=torch.cfloat, device=dev
)  # The naming nframes is because I'm essentially recording a certain number of snapshots

for _ in range(
    pars["prerun"]
):  # The prerun is for getting the system to a stable state before data is recorded
    gpsim.step()

for i in range(pars["nframes"]):
    gpsim.step()
    dispersion[i, :] = gpsim.psi[
        samplesY // 2 - 1, :
    ]  # Here I'm recording an x-axis of data to then get a dispersion relation

rdata = tnormSqr(gpsim.psi).real.detach().cpu().numpy()
kdata = tnormSqr(gpsim.psik).real.detach().cpu().numpy()
Emax = hbar * np.pi / pars["dt"]
extentE = [-kxmax, kxmax, 0, Emax / 4]

rdata = rdata[
    samplesY // 6 - 1 : samplesY - samplesY // 6,
    samplesX // 6 - 1 : samplesX - samplesX // 6,
]  # cropping the r-space data
extentr = [2 * startX / 3, 2 * endX / 3, 2 * startY / 3, 2 * endY / 3]
basedir = f"graphs/{datestamp}"
Path(basedir).mkdir(parents=True, exist_ok=True)
imshowBoilerplate(
    rdata,
    os.path.join(basedir, "exampler.pdf"),
    extent=extentr,
    xlabel="x [µm]",
    ylabel="y [µm]",
    aspect="equal",
    title=r"$|\psi_r|^2$",
)
startky = samplesY // 3 - 1
endky = samplesY - samplesY // 3
startkx = samplesX // 3 - 1
endkx = samplesX - samplesX // 3
kdata = kdata[startky:endky, startkx:endky]
extentk = [-kxmax / 3, kxmax / 3, -kymax / 3, kymax / 3]
imshowBoilerplate(
    kdata,
    os.path.join(basedir, "examplek.pdf"),
    extent=extentk,
    xlabel="$k_x$ [µm]",
    ylabel="$k_y$ [µm]",
    aspect="equal",
    title=r"$|\psi_k|^2$",
)

imshowBoilerplate(
    np.log(kdata + np.exp(-20)),
    os.path.join(basedir, "exampleklog.pdf"),
    extent=extentk,
    xlabel=r"$k_x$ [µ$m^{-1}$]",
    ylabel=r"$k_y$ [µ$m^{-1}$]",
    aspect="equal",
    title=r"$\ln(|\psi_k|^2 + e^{-20})",
)

dispersion = fftshift(fft(ifft(dispersion, axis=0), axis=1))
disp = tnormSqr(dispersion).real.detach().cpu().numpy()
start = disp.shape[0] // 2 - 1
end = start + disp.shape[0] // 8
disp = disp[start:end, :]
imshowBoilerplate(
    disp,
    os.path.join(basedir, "dispersion.pdf"),
    extent=extentE,
    xlabel="$k_x$ [µ$m^{-1}$]",
    ylabel="E [meV]",
    title=r"$I(k_x, E)$",
)

chime.theme("sonic")
chime.success()
