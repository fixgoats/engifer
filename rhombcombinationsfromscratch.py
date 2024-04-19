import argparse
import os
import shutil
import tomllib
from pathlib import Path
from time import gmtime, strftime

import chime
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from numpy.fft import fft, fftshift, ifft
from numpy.linalg import norm
from scipy.signal import convolve2d

from src.penrose import filterByRadius, goldenRatio, makeSunGrid
from src.solvers import SsfmGPGPU, hbar, npnormSqr, smoothnoise

plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"

now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
basedir = os.path.join("graphs", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--use-cached", action="store_true")
parser.add_argument("--check-penrose", action="store_true")
args = parser.parse_args()
if args.config is None and args.use_cached is None:
    exit("Need to specify config")

with open(f"{args.config}", "rb") as f:
    pars = tomllib.load(f)

shutil.copy(args.config, basedir)
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
ndistances = pars["ndistances"]
rhomblength0 = pars["rhomblength0"]
rhomblength1 = pars["rhomblength1"]
kind = pars["kind"]
rs = np.linspace(rhomblength0, rhomblength1, ndistances)

siminfo = (
    f'p{pars["pumpStrength"]}'
    f'n{pars["samplesX"]}'
    f's{pars["sigma"]}dt{dt}'
    f"xy{endX-startX}smallRbigeta"
)


def figBoilerplate():
    plt.cla()
    fig, ax = plt.subplots()
    fig.dpi = 300
    fig.figsize = (6.4, 4.8)
    return fig, ax


def imshowBoilerplate(data, filename, xlabel, ylabel, extent, title="", aspect="auto"):
    fig, ax = figBoilerplate()
    im = ax.imshow(
        data, aspect=aspect, origin="lower", interpolation="none", extent=extent
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    name = f"{filename}{siminfo}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(f"{basedir}/{name}.pdf")
    plt.close()
    print(f"Made plot {name} in {basedir}")


def gauss(x, y, sigmax, sigmay):
    return torch.exp(-x * x / sigmax - y * y / sigmay)


def npgauss(x, y, sigmax, sigmay):
    return np.exp(-x * x / sigmax - y * y / sigmay)


basis = np.array([[np.cos(i * np.pi / 5), np.sin(i * np.pi / 5)] for i in range(10)])

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

penrose = makeSunGrid(rhomblength0 * goldenRatio**4, 4)

if "cutoff" in pars:
    penrose = filterByRadius(penrose, pars["cutoff"])

if args.check_penrose:
    psample = penrose[0]
    minsep = 100
    for p in penrose[1:]:
        d = norm(p - psample)
        if d < minsep:
            minsep = d
    print(f"rhomblength is: {minsep:.3f}")
    print(f"radius is: {np.max(penrose[:,0])}")
    exit()

setupdict = {
    "thin": thin,
    "thick": thick,
    "thinthin": thinthin,
    "thickthin": thickthin,
    "thickthick": thickthick,
    "thinthickthin": thinthickthin,
    "penrose": penrose,
}

bleh = np.ndarray((nframes, ndistances))
if args.use_cached is False:
    cuda = torch.device("cuda")
    x = np.arange(startX, endX, dx).astype(np.complex64)
    gridY, gridX = np.meshgrid(x, x, indexing="ij")
    kxmax = np.pi / dx
    kymax = np.pi / dy
    dampingscale = endX * endX / 1.6
    damping = np.cosh((gridX * gridX + gridY * gridY) / dampingscale) - 1
    imshowBoilerplate(
        damping.real, "dampingpotential", "x", "y", [startX, endX, startY, endY]
    )
    dkx = 2 * kxmax / samplesX

    xv = torch.from_numpy(gridX)
    yv = torch.from_numpy(gridY)
    constV = -0.5j * (
        pars["gammalp"] * torch.ones((samplesY, samplesX)) + torch.from_numpy(damping)
    )

    npolarsgpu = torch.zeros((nframes), device="cuda")
    for j, r in enumerate(rs):
        psi = torch.from_numpy(smoothnoise(gridX, gridY))
        nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
        pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
        points = setupdict[kind]
        if kind == "penrose":
            for p in points * r / rhomblength0:
                pump += pars["pumpStrength"] * gauss(
                    xv - p[0], yv - p[1], pars["sigma"], pars["sigma"]
                )
        else:
            for p in r * points:
                pump += pars["pumpStrength"] * gauss(
                    xv - p[0], yv - p[1], pars["sigma"], pars["sigma"]
                )
        gpsim = SsfmGPGPU(
            dev=cuda,
            gridX=xv,
            gridY=yv,
            psi0=psi,
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
        extentr = np.array([startX, endX, startY, endY])
        extentk = np.array([-kxmax / 2, kxmax / 2, -kymax / 2, kymax / 2])
        spectrumgpu = torch.zeros((nframes), dtype=torch.cfloat, device="cuda")

        for _ in range(prerun):
            gpsim.step()

        for i in range(nframes):
            gpsim.step()
            npolarsgpu[i] = torch.sum(gpsim.psiNormSqr())
            spectrumgpu[i] = torch.sum(gpsim.psi)

        spectrum = (
            torch.fft.fftshift(torch.fft.ifft(spectrumgpu)).detach().cpu().numpy()
        )
        npolars = npolarsgpu.detach().cpu().numpy()
        np.save(f"tmp/spectrum{r:.3f}.npy", spectrum)
        bleh[:, j] = npnormSqr(spectrum) / np.max(npnormSqr(spectrum))
        fig, ax = figBoilerplate()
        ax.plot(dt * np.arange(nframes), npolars * dx * dy)
        name = f"{kind}nr{r:.3f}{siminfo}"
        plt.savefig(os.path.join(basedir, f"{name}.pdf"))
        plt.close()
        print(f"Made plot {name}")
        rdata = gpsim.psi.cpu().detach().numpy()
        imshowBoilerplate(
            npnormSqr(rdata),
            filename=f"{kind}rr{r:.3f}",
            xlabel="x (µm)",
            ylabel=r"y (µm)",
            title=r"$|\psi_r|^2$",
            extent=[startX, endX, startY, endY],
            aspect="equal",
        )
        kdata = gpsim.psik.cpu().detach().numpy()
        kdata = fftshift(kdata)[
            samplesY // 4 - 1 : samplesY - samplesY // 4,
            samplesX // 4 - 1 : samplesX - samplesX // 4,
        ]
        imshowBoilerplate(
            npnormSqr(kdata),
            filename=f"{kind}kr{r:.3f}",
            xlabel="$k_x$ (µ$m^{-1}$)",
            ylabel=r"$k_y$ (µ$m^{-1}$)",
            title=r"$|\psi_k|^2$",
            extent=[-kxmax / 2, kxmax / 2, -kymax / 2, kymax / 2],
            aspect="equal",
        )
        imshowBoilerplate(
            np.log(npnormSqr(kdata) + np.exp(-20)),
            filename=f"{kind}klogr{r:.3f}",
            xlabel="$k_x$ (µ$m^{-1}$)",
            ylabel=r"$k_y$ (µ$m^{-1}$)",
            title=r"$\ln(|\psi_k|^2 + e^{-20})$",
            extent=[-kxmax / 2, kxmax / 2, -kymax / 2, kymax / 2],
            aspect="equal",
        )
else:
    for i in rs:
        rhomblength = i
        sumd = np.load(f"tmp/intensity{rhomblength:.1f}.npy")
        bleh[:, i] = sumd

ommax = hbar * np.pi / dt
imshowBoilerplate(
    bleh[int(nframes * 0.5) : int(0.55 * nframes), ::-1],
    filename=f"{kind}intensityr{rhomblength0:.1f}",
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$I(E, d)$",
    extent=[rhomblength1, rhomblength0, 0, ommax / 10],
)

imshowBoilerplate(
    np.log(bleh[int(nframes * 0.5) : int(0.55 * nframes), ::-1]),
    filename=f"{kind}intensitylogr{rhomblength0:.1f}",
    xlabel="d (rhombii side length) (µm)",
    ylabel=r"E (meV)",
    title=r"$\ln(I(E, d))$",
    extent=[rhomblength1, rhomblength0, 0, ommax / 10],
)

chime.theme("sonic")
chime.success()
