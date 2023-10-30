import numpy as np
import torch
from src.solvers import TestingSsfmGPCUDA
from src.penrose import makeSunGrid, goldenRatio
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from matplotlib import animation
import tomllib
import argparse

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
if args.config is None:
    exit('Need to specify config')

with open(f'{args.config}', 'rb') as f:
    pars = tomllib.load(f)


def gauss(x, y, sigmax, sigmay):
    return torch.exp(-x * x / sigmax - y * y / sigmay)


def normSqr(x):
    return x.conj() * x


def npnormSqr(x):
    return x.real*x.real + x.imag*x.imag


cuda = torch.device('cuda')
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
gridY, gridX = torch.meshgrid(y, x, indexing='ij')
kxmax = np.pi / dx
kymax = np.pi / dy
dkx = 2 * kxmax / samplesX
psi = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
for i in range(60):
    psi += torch.exp(1j*(i-15)*gridX/6)

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
gpsim = TestingSsfmGPCUDA(psi0=psi,
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
                          dt=pars["dt"])

nframes = 512
extentr = [startX, endX, startY, endY]
extentk = [-kxmax, kxmax, -kymax, kymax]
bleh = np.zeros((nframes, samplesX), dtype=complex)

for i in range(nframes):
    gpsim.step()
    bleh[i, :] = gpsim.psi[127, :].detach().cpu().numpy()

fig, ax = plt.subplots()
fig.dpi = 300
fig.figsize = (6.4, 3.6)
rdata = gpsim.psi.detach().cpu().numpy()
im = ax.imshow(npnormSqr(rdata), origin='lower',
               extent=extentr)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
path = 'graphs/controlrplot.pdf'
ax.set_title(r'$|\psi_r|^2$')
ax.set_xlabel(r'x ($\mu$m)')
ax.set_ylabel(r'y ($\mu$m)')
plt.savefig(path)

plt.cla()
fig, ax = plt.subplots()
kdata = gpsim.psik.detach().cpu().numpy()
kdata = kdata
im = ax.imshow(npnormSqr(fftshift(kdata)), origin='lower',
               extent=extentk)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
path = 'graphs/controlkplot.pdf'
ax.set_title(r'$\ln(|\psi_k|^2 + 1)$')
ax.set_xlabel(r'$k_x$ ($\hbar/\mu$ m)')
ax.set_ylabel(r'$k_y$ ($\hbar/\mu$ m)')
plt.savefig(path)

Emax = np.pi / pars['dt']
bleh = fftshift(fft(ifft(bleh, axis=0), axis=1))
plt.cla()
fig, ax = plt.subplots()
im = ax.imshow(npnormSqr(bleh),
               aspect='auto',
               origin='lower',
               extent=[-kxmax, kxmax, -Emax, Emax])
ax.plot([-kxmax, kxmax], [25, 25], linewidth=1, color="red")
ax.plot([5, 5], [-Emax, Emax], linewidth=1, color="red")
path = 'graphs/controldispersion.pdf'
ax.set_title('Dispersion relation')
plt.savefig(path)
