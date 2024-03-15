import numpy as np
from numpy.linalg import norm
import torch
from src.solvers import SsfmGPGPU, npnormSqr, smoothnoise
from src.penrose import makeSunGrid, goldenRatio
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
import tomllib
import argparse
from pathlib import Path
from time import strftime, gmtime
import os

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

siminfo = f'p{pars["pumpStrength"]}'\
       f'n{pars["samplesX"]}'\
       f's{pars["sigma"]}dt{dt}'\
       f'xy{endX-startX}full'


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


cuda = torch.device('cuda')
x = torch.arange(startX, endX, dx).type(dtype=torch.cfloat)
y = torch.arange(startY, endY, dy).type(dtype=torch.cfloat)
nx = x.numpy()
nxv, nyv = np.meshgrid(nx, nx)
gridY, gridX = torch.meshgrid(y, x, indexing='ij')
kxmax = np.pi / dx
kymax = np.pi / dy
dkx = 2 * kxmax / samplesX
psi = torch.from_numpy(smoothnoise(nxv, nyv))

# constV = ((gridX / 50)**2 + (gridY / 50)**2)**8 - 0.5j*pars["gammalp"]
constV = -0.5j*pars["gammalp"]*torch.ones((samplesY, samplesX))

orgpoints = makeSunGrid(150., 4)
pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
for p in orgpoints:
    pump += pars["pumpStrength"]*gauss(gridX - p[0],
                                       gridY - p[1],
                                       pars["sigma"],
                                       pars["sigma"])

nR = torch.clone(pump)  # torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
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

nx = np.zeros(prerun + 1)
npolars = np.zeros(prerun+1)
npolars[0] = gpsim.psiNormSqr().cpu().detach().numpy().sum()
nx[0] = gpsim.nR.cpu().detach().numpy().sum()
angmom = np.zeros(prerun + 1)
for k in range(prerun):
    gpsim.step()
    npolars[k + 1] = gpsim.psiNormSqr().cpu().detach().numpy().sum()
    nx[k + 1] = gpsim.nR.cpu().detach().numpy().sum()
    angmom[k + 1] = gpsim.angmom().real.cpu()

rdata = gpsim.psi.cpu().detach().numpy()
imshowBoilerplate(npnormSqr(rdata),
                  filename=r"P3r",
                  xlabel=r"x (µm)",
                  ylabel=r"y (µm)",
                  title=r"$|\psi_r|^2$",
                  extent=[startX, endX, startY, endY])

kdata = gpsim.psik.cpu().detach().numpy()
kdata = fftshift(kdata)[samplesY // 3 - 1:samplesY - samplesY // 3,
                        samplesX // 3 - 1:samplesX - samplesX // 3]
imshowBoilerplate(np.log(npnormSqr(kdata) + np.exp(-10)),
                  filename="P3k",
                  xlabel=r"$k_x$ (µ$m^{-1}$)",
                  ylabel=r"$k_y$ (µ$m^{-1}$)",
                  title=r"$\ln(|\psi_k|^2 + e^{-20})$",
                  extent=[-kxmax/3, kxmax/3, -kymax/3, kymax/3])

t = dt*np.arange(prerun+1)
fig, ax = figBoilerplate()
ax.plot(t, npolars)
plt.savefig(f"{basedir}/npolars.pdf")
plt.close()

fig, ax = figBoilerplate()
ax.plot(t, nx)
plt.savefig(f"{basedir}/nx.pdf")
plt.close()

fig, ax = figBoilerplate()
ax.plot(t, angmom)
plt.savefig(f"{basedir}/angmom.pdf")
plt.close()
