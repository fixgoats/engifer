import numpy as np
import torch
from src.solvers import SsfmGPCUDA
from src.penrose import makeSunGrid, goldenRatio
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
import tomllib
import argparse

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

# parser = argparse.ArgumentParser()
# parser.add_argument('-o', '--output')
# args = parser.parse_args()
# if args.output is None:
#     exit('Need to specify filename for graph')

parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
if args.config is None:
    exit('Need to specify config')

with open(f'{args.config}', 'rb') as f:
    pars = tomllib.load(f)

def gauss(x, y, sigmax, sigmay):
    return torch.exp(-x * x / sigmax - y * y / sigmay)


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
psi = 0.1*torch.rand((samplesY, samplesX), dtype=torch.cfloat)

# Coefficients for GP equation
alpha = pars['alpha']
gammalp = pars['gammalp']
Gamma = pars['Gamma']
G = pars['G']
R = pars['R']
eta = pars['eta']
constV = ((gridX / 50)**2 + (gridY / 50)**2)**8 - 0.5j*gammalp


def normSqr(x):
    return x.conj() * x


radius = pars['radius']
divisions = pars['divisions']
pumpStrength = pars['pumpStrength']
sigma = pars['sigma']
points = makeSunGrid(radius, divisions)
minsep = pars["radius"] / (goldenRatio**pars["divisions"])
pars["minsep"] = minsep
pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
for p in points:
    pump += pumpStrength*gauss(gridX - p[0], gridY - p[1], sigma, sigma)

nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
gpsim = SsfmGPCUDA(psi0=psi,
                   gridX=gridX,
                   gridY=gridY,
                   m=pars['m'],
                   nR0=nR,
                   alpha=alpha,
                   Gamma=Gamma,
                   gammalp=gammalp,
                   R=R,
                   pump=pump,
                   G=G,
                   eta=eta,
                   constV=constV,
                   dt=pars['dt'])

for _ in range(3000):
    gpsim.step()

fig, ax = plt.subplots()
fig.dpi = 300
fig.figsize = (6.4, 3.6)
extentr = [startX, endX, startY, endY]
extentk = [-kxmax/2, kxmax/2, -kymax/2, kymax/2]
nppsi = normSqr(gpsim.psi).real.cpu().detach().numpy()
im = ax.imshow(nppsi,
               origin='lower',
               extent=extentr)
ax.set_title('$|\\psi_r|^2$')
ax.set_xlabel(r'x ($\mu$m)')
ax.set_ylabel(r'y ($\mu$m)')
vmax = np.max(nppsi)
vmin = np.min(nppsi)
ax.set_clim(vmin, vmax)
plt.colorbar(im, ax=ax)
positions = ax.scatter(points[:, 0],
                       points[:, 1],
                       s=0.5,
                       linewidths=1,
                       color='#ff6347')
plt.savefig(f'rpframer{radius}d{divisions}p{pumpStrength}s{sigma}.pdf')
plt.cla()
fig, ax = plt.subplots()
fig.dpi = 300
fig.figsize = (6.4, 3.6)
psik0 = normSqr(gpsim.psik).real.cpu().detach().numpy()
im = ax.imshow(np.log(fftshift(psik0)[255:1024-256, 255:1024-256] + 1),
               origin='lower',
               extent=extentk)
ax.set_title('$|\\psi_k|^2$')
ax.set_xlabel(r'$k_x$ ($\hbar/\mu$m)')
ax.set_ylabel(r'$k_y$ ($\hbar/\mu$m)')
plt.colorbar(im, ax=ax)
plt.savefig(f'kpframer{radius}d{divisions}p{pumpStrength}s{sigma}.pdf')
