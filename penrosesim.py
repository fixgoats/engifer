import numpy as np
import torch
from src.solvers import SsfmGPGPU, npnormSqr, hbar
from src.penrose import makeSunGrid, goldenRatio
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

siminfo = f'r{pars["radius"]}'\
       f'd{pars["divisions"]}'\
       f'p{pars["pumpStrength"]}'\
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
    print(f'Made plot {name} in {basedir}')


if args.use_cached is False:
    cuda = torch.device('cuda')
    dx = (endX - startX) / samplesX
    dy = (endY - startY) / samplesY
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

    points = makeSunGrid(pars["radius"], pars["divisions"])
    minsep = pars["radius"] / (goldenRatio**pars["divisions"])
    pars["minsep"] = minsep
    pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
    for p in points:
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

    fps = 24
    fig, [ax1, ax2] = plt.subplots(1, 2)
    fig.dpi = 300
    fig.figsize = (6.4, 3.6)
    extentr = np.array([startX, endX, startY, endY])
    extentk = np.array([-kxmax/2, kxmax/2, -kymax/2, kymax/2])
    dispersion = np.zeros((nframes, samplesX), dtype=complex)
    npolars = np.zeros(nframes + prerun)

    for i in range(prerun):
        gpsim.step()
        rdata = gpsim.psi.detach().cpu().numpy()
        npolars[i] = np.sum(npnormSqr(rdata))

    for i in range(nframes):
        gpsim.step()
        rdata = gpsim.psi.detach().cpu().numpy()
        npolars[i + prerun] = np.sum(npnormSqr(rdata))
        dispersion[i, :] = rdata[samplesY//2 - 1, :]

    npolars = npolars * dx * dy
    Path("tmp").mkdir(parents=True, exist_ok=True)
    rdata = gpsim.psi.detach().cpu().numpy()
    np.save("tmp/rdata.npy", rdata)
    kdata = gpsim.psik.detach().cpu().numpy()
    np.save("tmp/kdata.npy", kdata)
    np.save("tmp/disp.npy", dispersion)
    np.save("tmp/extentr.npy", extentr)
    np.save("tmp/extentk.npy", extentk)
    np.save("tmp/points.npy", points)
    Emax = hbar * np.pi / pars['dt']
    extentE = [-kxmax, kxmax, 0, Emax]
    np.save("tmp/npolars.npy", npolars)
    np.save("tmp/extentE.npy", extentE)


else:
    rdata = np.load("tmp/rdata.npy")
    kdata = np.load("tmp/kdata.npy")
    dispersion = np.load("tmp/disp.npy")
    extentr = np.load("tmp/extentr.npy")
    extentk = np.load("tmp/extentk.npy")
    extentE = np.load("tmp/extentE.npy")
    points = np.load("tmp/points.npy")
    npolars = np.load("tmp/npolars.npy")
    samplesY, samplesX = np.shape(rdata)

imshowBoilerplate(npnormSqr(rdata),
                  filename="penroser",
                  xlabel=r'x (µm)',
                  ylabel=r'y (µm)',
                  extent=extentr,
                  title='$|\\psi_r|^2$')

kdata = npnormSqr(fftshift(kdata)[samplesY//4 - 1:samplesY - samplesY // 4,
                                  samplesX//4 - 1:samplesX - samplesX//4])
imshowBoilerplate(kdata,
                  filename="penrosek",
                  xlabel=r'$k_x$ ($\mu m^{-1}$)',
                  ylabel=r'$k_y$ ($\mu m^{-1}$)',
                  extent=extentk,
                  title=r'$|\psi_k|^2$')

imshowBoilerplate(np.log(kdata+np.exp(-10)),
                  filename='penroseklog',
                  xlabel=r'$k_x$ ($\mu m^{-1}$)',
                  ylabel=r'$k_y$ ($\mu m^{-1}$)',
                  extent=extentk,
                  title=r'$\ln(|\psi_k|^2 + e^{-10})$')

dispersion = fftshift(fft(ifft(dispersion, axis=0), axis=1))
if not pars["wneg"]:
    start = dispersion.shape[0] // 2 - 1
    dispersion = dispersion[start:, :]

imshowBoilerplate(np.sqrt(npnormSqr(dispersion)),
                  filename='penrosedisp',
                  xlabel=r'$k_x$ ($\mu m^{-1}$)',
                  ylabel=r'$E$ (meV)',
                  extent=extentk,
                  title=r'$\rho(E, k_x)$')

imshowBoilerplate(np.log(np.sqrt(npnormSqr(dispersion)) + np.exp(-10)),
                  filename='penrosedisplog',
                  xlabel=r'$k_x$ ($\mu m^{-1}$)',
                  ylabel=r'$E$ (meV)',
                  extent=extentk,
                  title=r'$\ln(\rho(E, k_x)$ + e^{-10})')

fig, ax = figBoilerplate()
ax.plot(dt * np.arange(prerun + nframes), npolars)
ax.set_xlabel("t (ps)")
ax.set_ylabel("Number of polaritons")
name = f'penrosen{siminfo}'
plt.savefig(f'{basedir}/{name}.pdf')
print(f'Made number plot {name} in {basedir}')
