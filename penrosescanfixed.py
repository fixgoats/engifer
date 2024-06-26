import argparse
import os
import tomllib
from pathlib import Path
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from numpy.fft import fft, fftshift, ifft

from src.penrose import goldenRatio, makeSunGrid
from src.solvers import SsfmGPGPU, hbar, npnormSqr

plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"

now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
basedir = os.path.join("graphs", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--use-cached", action="store_true")
args = parser.parse_args()
if args.config is None and args.use_cached is None:
    exit("Need to specify config")

with open(f"{args.config}", "rb") as f:
    pars = tomllib.load(f)


def gauss(x, y, sigmax, sigmay):
    return torch.exp(-x * x / sigmax - y * y / sigmay)


endX = pars["endX"]
startX = pars["startX"]
samplesX = pars["samplesX"]
endY = pars["endY"]
startY = pars["startY"]
samplesY = pars["samplesY"]
dt = pars["dt"]
nframes = pars["nframes"]
prerun = pars["prerun"]
dx = (endX - startX) / samplesX
dy = (endY - startY) / samplesY
ndistances = 14
rhomblength0 = 24
rhomblength1 = 10
rad0 = rhomblength0 * goldenRatio**4
rad1 = rhomblength1 * goldenRatio**4
actualrad = 110
radoffset = rad0 - actualrad
dr = (rad0 - rad1) / ndistances
rs = np.arange(rad0, rad1, -dr)
intermediaterun = pars["intermediaterun"]

siminfo = (
    f'd{pars["divisions"]}'
    f'p{pars["pumpStrength"]}'
    f'n{pars["samplesX"]}'
    f's{pars["sigma"]}dt{dt}'
    f"xy{endX-startX}smallRbigeta"
)


def figBoilerplate():
    plt.cla()
    fig, ax = plt.subplots()
    fig.dpi = 300
    fig.figsize = (6.4, 3.6)
    return fig, ax


def imshowBoilerplate(data, filename, xlabel, ylabel, extent, title=""):
    fig, ax = figBoilerplate()
    im = ax.imshow(
        data, aspect="auto", origin="lower", interpolation="none", extent=extent
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    name = f"{filename}{siminfo}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(f"{basedir}/{name}.pdf")
    plt.close()
    print(f"Made plot {name} in {basedir}")


def filterByRadius(array, r):
    returnarray = np.ndarray((0, 2))
    for row in array:
        if row[0] ** 2 + row[1] ** 2 < r**2:
            returnarray = np.append(returnarray, [row], axis=0)
    return returnarray


bleh = np.ndarray((nframes, ndistances))
if args.use_cached is False:
    cuda = torch.device("cuda")
    x = torch.arange(startX, endX, dx)
    y = torch.arange(startY, endY, dy)
    x = x.type(dtype=torch.cfloat)
    y = y.type(dtype=torch.cfloat)
    gridY, gridX = torch.meshgrid(y, x, indexing="ij")
    kxmax = np.pi / dx
    kymax = np.pi / dy
    dkx = 2 * kxmax / samplesX
    psi = 0.1 * torch.rand((samplesY, samplesX), dtype=torch.cfloat)

    # constV = ((gridX / 50)**2 + (gridY / 50)**2)**8 - 0.5j*pars["gammalp"]
    constV = -0.5j * pars["gammalp"] * torch.ones((samplesY, samplesX))

    points = filterByRadius(makeSunGrid(rs[0], 4), actualrad)
    print(points.shape)
    pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
    for p in points:
        pump += pars["pumpStrength"] * gauss(
            gridX - p[0], gridY - p[1], pars["sigma"], pars["sigma"]
        )

    nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
    gpsim = SsfmGPGPU(
        dev=cuda,
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
        dt=dt,
    )
    npolars = np.zeros(nframes)
    for k in range(prerun):
        gpsim.step()

    for j, r in enumerate(rs):
        newpoints = ((r - radoffset) / actualrad) * points
        pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
        # psi = 0.1*torch.rand((samplesY, samplesX), dtype=torch.cfloat)
        # nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
        for p in newpoints:
            pump += pars["pumpStrength"] * gauss(
                gridX - p[0], gridY - p[1], pars["sigma"], pars["sigma"]
            )
        gpsim.pump = pump.to(cuda)
        # gpsim.psi = psi.to(cuda)
        # gpsim.nR = nR.to(cuda)
        extentr = np.array([startX, endX, startY, endY])
        extentk = np.array([-kxmax / 2, kxmax / 2, -kymax / 2, kymax / 2])
        dispersion = np.zeros((nframes, samplesY, samplesX), dtype=complex)

        for k in range(intermediaterun):
            gpsim.step()

        for i in range(nframes):
            gpsim.step()
            rdata = gpsim.psi.detach().cpu().numpy()
            npolars[i] = np.sum(npnormSqr(rdata))
            dispersion[i, :] = rdata

        dispersion = fftshift(ifft(dispersion, axis=0, norm="ortho"), axes=0)
        sumd = (
            np.sum(npnormSqr(dispersion), axis=(1, 2))
            * dx
            * dy
            / ((endX - startX) * (endY - startY))
        )
        rhomblength = r / goldenRatio**4
        np.save(f"tmp/intensity{rhomblength:.0f}.npy", sumd)
        bleh[:, j] = sumd
        fig, ax = figBoilerplate()
        ax.plot(dt * np.arange(nframes), npolars * dx * dy)
        name = f"penrosenr{rhomblength:.1f}{siminfo}"
        plt.savefig(os.path.join(basedir, f"{name}.pdf"))
        plt.close()
        print(f"Made plot {name}")
        rdata = gpsim.psi.cpu().detach().numpy()
        imshowBoilerplate(
            npnormSqr(rdata),
            filename=f"penroserr{rhomblength:.1f}",
            xlabel="x (µm)",
            ylabel=r"y (µm)",
            title=r"$|\psi_r|^2$",
            extent=[startX, endX, startY, endY],
        )
        kdata = gpsim.psik.cpu().detach().numpy()
        kdata = fftshift(kdata)[
            samplesY // 4 - 1 : samplesY - samplesY // 4,
            samplesX // 4 - 1 : samplesX - samplesX // 4,
        ]
        imshowBoilerplate(
            npnormSqr(kdata),
            filename=f"penrosekr{rhomblength:.1f}",
            xlabel="$k_x$ (µ$m^{-1}$)",
            ylabel=r"$k_y$ (µ$m^{-1}$)",
            title=r"$|\psi_k|^2$",
            extent=[-kxmax / 2, kxmax / 2, -kymax / 2, kymax / 2],
        )
        imshowBoilerplate(
            np.log(npnormSqr(kdata) + np.exp(-20)),
            filename=f"penroseklogr{rhomblength:.1f}",
            xlabel="$k_x$ (µ$m^{-1}$)",
            ylabel=r"$k_y$ (µ$m^{-1}$)",
            title=r"$\ln(|\psi_k|^2 + e^{-20})$",
            extent=[-kxmax / 2, kxmax / 2, -kymax / 2, kymax / 2],
        )


else:
    for i in rs:
        rhomblength = i / goldenRatio**4
        sumd = np.load(f"tmp/intensity{rhomblength:.1f}.npy")
        bleh[:, i] = sumd

ommax = hbar * np.pi / dt
imshowBoilerplate(
    bleh[nframes // 2 - 1 : nframes - nframes // 4, ::-1],
    filename=f"penroseintensityr{actualrad:.1f}",
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$I(E, d)$",
    extent=[rhomblength1, rhomblength0, 0, ommax / 2],
)
imshowBoilerplate(
    np.log(bleh[nframes // 2 - 1 : nframes - nframes // 4, ::-1]),
    filename=f"penroseintensitylogr{actualrad:.1f}",
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$\ln(I(E, d))$",
    extent=[rhomblength1, rhomblength0, 0, ommax / 2],
)
