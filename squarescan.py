import numpy as np
import torch
from src.solvers import SsfmGPGPU, npnormSqr, hbar
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from matplotlib import animation
import tomllib
import argparse
from pathlib import Path
from datetime import date

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

datestamp = date.today()
basedir = f"graphs/{datestamp}"
Path(basedir).mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--use-cached', action='store_true')
args = parser.parse_args()
if args.config is None and args.use_cached is None:
    exit('Need to specify config')

with open(f'{args.config}', 'rb') as f:
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
ndistances = 16
sidelength0 = 15
sidelength1 = 7
dside = (sidelength1 - sidelength0) / ndistances
sidelengths = np.arange(sidelength0, sidelength1, dside)
intermediaterun = pars["intermediaterun"]

siminfo = f'p{pars["pumpStrength"]}'\
       f'n{pars["samplesX"]}'\
       f's{pars["sigma"]}dt{dt}'\
       f'xy{endX-startX}'


def figBoilerplate():
    plt.cla()
    fig, ax = plt.subplots()
    fig.dpi = 300
    fig.figsize = (6.4, 3.6)
    return fig, ax


def imshowBoilerplate(data, filename, xlabel, ylabel, extent, title=""):
    fig, ax = figBoilerplate()
    im = ax.imshow(data,
                   aspect='auto',
                   origin='lower',
                   interpolation='none',
                   extent=extent)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    name = f'{filename}{siminfo}'
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(f'{basedir}/{name}.pdf')
    plt.close()
    print(f'Made plot {name} in {basedir}')


bleh = np.ndarray((nframes, ndistances))
if args.use_cached is False:
    cuda = torch.device('cuda')
    x = torch.arange(startX, endX, dx)
    y = torch.arange(startY, endY, dy)
    x = x.type(dtype=torch.cfloat)
    y = y.type(dtype=torch.cfloat)
    gridY, gridX = torch.meshgrid(y, x, indexing='ij')
    kxmax = np.pi / dx
    kymax = np.pi / dy
    dkx = 2 * kxmax / samplesX
    psi = 0.1*torch.rand((samplesY, samplesX), dtype=torch.cfloat)

    # constV = ((gridX / 50)**2 + (gridY / 50)**2)**8 - 0.5j*pars["gammalp"]
    constV = -0.5j*pars["gammalp"]*torch.ones((samplesY, samplesX))

    squaregrid = np.array([[sidelength0 * i - 75, sidelength0 * j - 75] for i in range(10) for j in range(10)])
    pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
    for p in squaregrid:
        pump += pars["pumpStrength"]*gauss(gridX - p[0],
                                           gridY - p[1],
                                           pars["sigma"],
                                           pars["sigma"])

    nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
    gpsim = SsfmGPGPU(dev=cuda,
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
                      dt=dt)
    npolars = np.zeros(nframes)
    for k in range(prerun):
        gpsim.step()

    for j, d in enumerate(sidelengths):
        points = squaregrid * d / sidelength0
        pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
        for p in points:
            pump += pars["pumpStrength"]*gauss(gridX - p[0],
                                               gridY - p[1],
                                               pars["sigma"],
                                               pars["sigma"])
        gpsim.pump = pump.to(cuda)
        extentr = np.array([startX, endX, startY, endY])
        extentk = np.array([-kxmax/2, kxmax/2, -kymax/2, kymax/2])
        dispersion = np.zeros((nframes, samplesY, samplesX), dtype=complex)

        for k in range(intermediaterun):
            gpsim.step()

        for i in range(nframes):
            gpsim.step()
            rdata = gpsim.psi.detach().cpu().numpy()
            npolars[i] = np.sum(npnormSqr(rdata))
            dispersion[i, :] = rdata

        dispersion = fftshift(ifft(dispersion, axis=0, norm='ortho'), axes=0)
        sumd = np.sum(npnormSqr(dispersion), axis=(1, 2)) * dx * dy / ((endX - startX) * (endY - startY))
        np.save(f"tmp/intensity{d:.0f}.npy", sumd)
        bleh[:, j] = sumd
        fig, ax = figBoilerplate()
        ax.plot(dt * np.arange(nframes), npolars * dx * dy)
        name = f"squarenr{d:.1f}{siminfo}"
        plt.savefig(f"graphs/{datestamp}/{name}.pdf")
        plt.close()
        print(f"Made plot {name}")
        rdata = gpsim.psi.cpu().detach().numpy()
        imshowBoilerplate(npnormSqr(rdata),
                          filename=f"squarer{d:.1f}",
                          xlabel="x (µm)",
                          ylabel=r"y (µm)",
                          title=r"$|\psi_r|^2$",
                          extent=[startX, endX, startY, endY])
        kdata = gpsim.psik.cpu().detach().numpy()
        kdata = fftshift(kdata)[samplesY // 4 - 1:samplesY - samplesY // 4,
                                samplesX // 4 - 1:samplesX - samplesX // 4]
        imshowBoilerplate(npnormSqr(kdata),
                          filename=f"squarekr{d:.1f}",
                          xlabel="$k_x$ (µ$m^{-1}$)",
                          ylabel=r"$k_y$ (µ$m^{-1}$)",
                          title=r"$|\psi_k|^2$",
                          extent=[-kxmax/2, kxmax/2, -kymax/2, kymax/2])
        imshowBoilerplate(np.log(npnormSqr(kdata) + np.exp(-20)),
                          filename=f"squareklogr{d:.1f}",
                          xlabel="$k_x$ (µ$m^{-1}$)",
                          ylabel=r"$k_y$ (µ$m^{-1}$)",
                          title=r"$\ln(|\psi_k|^2 + e^{-20})$",
                          extent=[-kxmax/2, kxmax/2, -kymax/2, kymax/2])


else:
    for i in sidelengths:
        sumd = np.load(f"tmp/intensity{i:.1f}.npy")
        bleh[:, i] = sumd

ommax = hbar * np.pi / dt
imshowBoilerplate(bleh[nframes//2-1:],
                  filename=f"squareintensityr{sidelength0:.1f}",
                  xlabel="d (rhombii side length) (µm)",
                  ylabel=r"E (meV)",
                  title=r"$I(E, d)$",
                  extent=[sidelength0, sidelength1, 0, ommax])
imshowBoilerplate(np.log(bleh[nframes//2-1:]),
                  filename=f"squareintensitylogr{sidelength0:.1f}",
                  xlabel="d (rhombii side length) (µm)",
                  ylabel=r"E (meV)",
                  title=r"$\ln(I(E, d))$",
                  extent=[sidelength0, sidelength1, 0, ommax])
