import tomllib
import argparse
import os
from pathlib import Path
from time import strftime, gmtime

import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift
from scipy.signal import convolve2d
from matplotlib import animation

from src.solvers import SsfmGPGPU, npnormSqr, hbar
from src.penrose import goldenRatio

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

now = gmtime()
day = strftime('%Y-%m-%d', now)
timeofday = strftime('%H.%M', now)
basedir = os.path.join('graphs', day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--use-cached', action='store_true')
args = parser.parse_args()
if args.config is None and args.use_cached is None:
    exit('Need to specify config')

with open(f'{args.config}', 'rb') as f:
    pars = tomllib.load(f)


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
rs = np.linspace(rhomblength0, rhomblength1, 30)

siminfo = 'rhombus'\
       f'p{pars["pumpStrength"]}'\
       f'n{pars["samplesX"]}'\
       f's{pars["sigma"]}dt{dt}'\
       f'xy{endX-startX}smallRbigeta'


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


def gauss(x, y, sigmax, sigmay):
    return torch.exp(-x * x / sigmax - y * y / sigmay)


def smoothnoise(xv):
    random = np.random.uniform(-1, 1, np.shape(xv)) + 1j * np.random.uniform(-1, 1, np.shape(xv))
    krange = np.linspace(-2, 2, num=21)
    kbasex, kbasey = np.meshgrid(krange, krange)
    kernel = gauss(kbasex, kbasey, 1, 1)
    kernel /= np.sum(kernel)
    output = convolve2d(random, kernel, mode='same')
    return output / np.sqrt(np.sum(npnormSqr(output)))


rhombus = np.array([
    [0, 0],
    [np.cos(np.radians(36)), np.sin(np.radians(36))],
    [np.cos(np.radians(144)), np.sin(np.radians(144))],
    [0, 2*np.sin(np.radians(144))]
    ])

bleh = np.ndarray((nframes, ndistances))
if args.use_cached is False:
    cuda = torch.device('cuda')
    x = np.arange(startX, endX, dx).astype(np.complex64)
    gridY, gridX = np.meshgrid(x, x, indexing='ij')
    kxmax = np.pi / dx
    kymax = np.pi / dy
    dkx = 2 * kxmax / samplesX
    psi = 0.1*torch.from_numpy(smoothnoise(gridX))

    # constV = ((gridX / 50)**2 + (gridY / 50)**2)**8 - 0.5j*pars["gammalp"]
    constV = -0.5j*pars["gammalp"]*torch.ones((samplesY, samplesX))

    pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
    for p in rhomblength0*rhombus:
        pump += pars["pumpStrength"]*gauss(gridX - p[0],
                                           gridY - p[1],
                                           pars["sigma"],
                                           pars["sigma"])

    nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
    gpsim = SsfmGPGPU(dev=cuda,
                      psi0=psi,
                      gridX=torch.from_numpy(gridX),
                      gridY=torch.from_numpy(gridY),
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

    for j, r in enumerate(rs):
        pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
        newpoints = r * rhombus
        # psi = 0.1*torch.rand((samplesY, samplesX), dtype=torch.cfloat)
        # nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
        for p in newpoints:
            pump += pars["pumpStrength"]*gauss(gridX - p[0],
                                               gridY - p[1],
                                               pars["sigma"],
                                               pars["sigma"])
        gpsim.pump = pump.to(cuda)
        # gpsim.psi = psi.to(cuda)
        # gpsim.nR = nR.to(cuda)
        extentr = np.array([startX, endX, startY, endY])
        extentk = np.array([-kxmax/2, kxmax/2, -kymax/2, kymax/2])
        spectrum = np.zeros(nframes, dtype=complex)

        for _ in range(pars["intermediaterun"]):
            gpsim.step()

        for i in range(nframes):
            gpsim.step()
            rdata = gpsim.psi.detach().cpu().numpy()
            npolars[i] = np.sum(npnormSqr(rdata))
            spectrum[i] = np.sum(rdata)

        spectrum = fftshift(ifft(spectrum))
        np.save(f"tmp/intensity{r:.1f}.npy", spectrum)
        fig, ax = figBoilerplate()
        ax.plot(dt * np.arange(nframes), npolars * dx * dy)
        name = f"penrosenr{rhomblength:.1f}{siminfo}"
        plt.savefig(os.path.join(basedir, f'{name}.pdf'))
        plt.close()
        print(f"Made plot {name}")
        rdata = gpsim.psi.cpu().detach().numpy()
        imshowBoilerplate(npnormSqr(rdata),
                          filename=f"penroserr{rhomblength:.1f}",
                          xlabel="x (µm)",
                          ylabel=r"y (µm)",
                          title=r"$|\psi_r|^2$",
                          extent=[startX, endX, startY, endY])
        kdata = gpsim.psik.cpu().detach().numpy()
        kdata = fftshift(kdata)[samplesY // 4 - 1:samplesY - samplesY // 4,
                                samplesX // 4 - 1:samplesX - samplesX // 4]
        imshowBoilerplate(npnormSqr(kdata),
                          filename=f"penrosekr{rhomblength:.1f}",
                          xlabel="$k_x$ (µ$m^{-1}$)",
                          ylabel=r"$k_y$ (µ$m^{-1}$)",
                          title=r"$|\psi_k|^2$",
                          extent=[-kxmax/2, kxmax/2, -kymax/2, kymax/2])
        imshowBoilerplate(np.log(npnormSqr(kdata) + np.exp(-20)),
                          filename=f"penroseklogr{rhomblength:.1f}",
                          xlabel="$k_x$ (µ$m^{-1}$)",
                          ylabel=r"$k_y$ (µ$m^{-1}$)",
                          title=r"$\ln(|\psi_k|^2 + e^{-20})$",
                          extent=[-kxmax/2, kxmax/2, -kymax/2, kymax/2])


else:
    for i in rs:
        rhomblength = i / goldenRatio**4
        sumd = np.load(f"tmp/intensity{rhomblength:.1f}.npy")
        bleh[:, i] = sumd

ommax = hbar * np.pi / dt
imshowBoilerplate(bleh[nframes//2-1:nframes - nframes // 4, ::-1],
                  filename=f"penroseintensityr{actualrad:.1f}",
                  xlabel="d (rhombii side length) (µm)",
                  ylabel=r"E (meV)",
                  title=r"$I(E, d)$",
                  extent=[rhomblength1, rhomblength0, 0, ommax/2])
imshowBoilerplate(np.log(bleh[nframes//2-1:nframes - nframes // 4, ::-1]),
                  filename=f"penroseintensitylogr{actualrad:.1f}",
                  xlabel="d (rhombii side length) (µm)",
                  ylabel=r"E (meV)",
                  title=r"$\ln(I(E, d))$",
                  extent=[rhomblength1, rhomblength0, 0, ommax/2])
