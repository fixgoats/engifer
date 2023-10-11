import numpy as np
import torch
from src.solvers import SsfmGPCUDA
from src.penrose import makeSunGrid
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from matplotlib import animation
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
startX = -40  # micrometers
endX = 40
startY = -40
endY = 40
dx = (endX - startX) / samplesX
dy = (endY - startY) / samplesY
x = torch.arange(startX, endX, dx)
y = torch.arange(startY, endY, dy)
x = x.type(dtype=torch.cfloat)
y = y.type(dtype=torch.cfloat)
gridY, gridX = torch.meshgrid(y, x, indexing='ij')
kxmax = np.pi / dx
kymax = np.pi / dy
dt = 0.07
m = 0.5
psi = 0.1*torch.sin(gridX)*torch.cos(gridY)

# Coefficients for GP equation
alpha = 0.1
gammalp = 2
Gamma = 1
G = 0.1
R = 2
eta = 1
constV = ((gridX / 25)**2 + (gridY / 25)**2)**8 - 0.5j*gammalp


def normSqr(x):
    return x.conj() * x


points = makeSunGrid(20, 5)
pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
for p in points:
    pump += 5*gauss(gridX - p.real, gridY - p.imag, 0.1, 0.1)
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

nframes = 2048
fps = 24
fig, ax = plt.subplots()
im = ax.imshow(normSqr(gpsim.psi).real.cpu().detach().numpy(),
               origin='lower',
               extent=[startX, endX, startY, endY])
ax.set_xlabel(r'x ($\mu$m)')
ax.set_ylabel(r'y ($\mu$m)')
fig.colorbar(im, ax=ax)
bleh = np.zeros((2048, 512), dtype=complex)


def init():
    return [im]


def animate_heatmap(frame):
    gpsim.step()
    data = gpsim.psi.detach().cpu().numpy()
    bleh[frame, :] = data[127, :]
    im.set_data(normSqr(data).real)
    # vmin = np.min(data)
    # vmax = np.max(data)
    im.set_clim(0, 5)
    return [im]


anim = animation.FuncAnimation(fig,
                               animate_heatmap,
                               init_func=init,
                               frames=nframes)
FFwriter = animation.FFMpegWriter(fps=fps,
                                  metadata={'copyright': 'Public Domain'})
anim.save('animations/lessintensequasicrystal.mp4', writer=FFwriter)

Emax = np.pi / dt
bleh = fftshift(fft(ifft(bleh, axis=0), axis=1))
plt.cla()
fig, ax = plt.subplots()
im = ax.imshow(np.log(normSqr(bleh).real),
               aspect='auto',
               extent=[-kxmax, kxmax, -Emax, Emax])
plt.savefig('dispersion.pdf')
