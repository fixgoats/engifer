import numpy as np
import torch
from src.solvers import SsfmGPCUDA
from src.penrosecoords import pgrid
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
#import argparse

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

# parser = argparse.ArgumentParser()
# parser.add_argument('-o', '--output')
# args = parser.parse_args()
# if args.output is None:
#     exit('Need to specify filename for graph')


def gauss(x, y, sigmax, sigmay):
    return torch.exp(-x * x / sigmax - y * y / sigmay)


cuda = torch.device('cuda')
samplesX = 512
samplesY = 512
startX = -80  # micrometers
endX = 80
startY = -80
endY = 80
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
dt = 0.07
m = 0.5
psi = 0.1*torch.rand((samplesY, samplesX), dtype=torch.cfloat)

# Coefficients for GP equation
alpha = 0.01
gammalp = 2
Gamma = 1
G = 0.1
R = 2
eta = 1
constV = ((gridX / 50)**2 + (gridY / 50)**2)**8 - 0.5j*gammalp


def normSqr(x):
    return x.conj() * x


radius = 10
divisions = 3
pumpStrength = 3.2
w = 1.5
points = pgrid(radius, divisions)
truer = np.max(np.sqrt(np.sum(points*points, axis=1)))
print(truer)
pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
pumpPos = np.zeros((samplesY, samplesX))
for p in points:
    addition = pumpStrength*gauss(gridX - p[0], gridY - p[1], w, w)
    pump += addition

nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
gpsim = SsfmGPCUDA(psi0=psi,
                   gridX=gridX,
                   gridY=gridY,
                   m=m,
                   nR0=nR,
                   alpha=alpha,
                   Gamma=Gamma,
                   gammalp=gammalp,
                   R=R,
                   pump=pump,
                   G=G,
                   eta=eta,
                   constV=constV,
                   dt=dt)

for _ in range(4000):
    gpsim.step()

fig, ax = plt.subplots()
fig.dpi = 300
fig.figsize = (6.4, 3.6)
extentr = [startX, endX, startY, endY]
extentk = [-kxmax/2, kxmax/2, -kymax/2, kymax/2]
im = ax.imshow(normSqr(gpsim.psi).real.cpu().detach().numpy(),
                origin='lower',
                extent=extentr)
ax.set_title('$|\\psi_r|^2$')
ax.set_xlabel(r'x ($\mu$m)')
ax.set_ylabel(r'y ($\mu$m)')
plt.colorbar(im, ax=ax)
positions = ax.scatter(points[:, 0],
                       points[:, 1],
                       s=0.5,
                       linewidths=1,
                       color='#ff6347')
plt.savefig(f'rpframer{radius}d{divisions}p{pumpStrength}w{w}.pdf')
plt.cla()
fig, ax = plt.subplots()
fig.dpi = 300
fig.figsize = (6.4, 3.6)
psik0 = normSqr(gpsim.psik).real.cpu().detach().numpy()
im = ax.imshow(fftshift(psik0)[191:512-192, 191:512-192],
               origin='lower',
               extent=extentk)
ax.set_title('$|\\psi_k|^2$')
ax.set_xlabel(r'$k_x$ ($\hbar/\mu$m)')
ax.set_ylabel(r'$k_y$ ($\hbar/\mu$m)')
plt.colorbar(im, ax=ax)
plt.savefig(f'kpframer{radius}d{divisions}p{pumpStrength}w{w}.pdf')
