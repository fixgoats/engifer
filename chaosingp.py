import numpy as np
import torch
from src.solvers import SsfmGPCUDA
from src.penrose import makeSunGrid
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


def exactsol(x, y, t, a, m):
    return (a / (a + 1.0j * t / m)) \
        * torch.exp(-(x * x + y * y) / (2 * (a + 1.0j * t / m)))


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
x = x.type(dtype=torch.cfloat)
gridY, gridX = torch.meshgrid(x, x, indexing='ij')
kxmax = np.pi / dx
kymax = np.pi / dy
dt = 0.05
m = 0.5
psi = torch.sin(gridX)*torch.cos(gridY)

# Coefficients for GP equation
alpha = 0.01
gammalp = 2
Gamma = 1
G = 0.1
R = 2
eta = 1
constV = ((gridX / 30)**2 + (gridY / 30)**2)**8 - 0.5j*gammalp


def normSqr(x):
    return x.conj() * x


points = makeSunGrid(20, 5)
pump1 = torch.zeros((samplesX, samplesY), dtype=torch.cfloat)
for p in points:
    pump1 += 100*gauss(gridX - p.real, gridY - p.imag, 0.1, 0.1)

pump2 = torch.zeros((samplesX, samplesY), dtype=torch.cfloat)
for p in points:
    if p != 0+0j:
        pump2 += 100*gauss(gridX - p.real, gridY - p.imag, 0.1, 0.1)
    else:
        pump2 += 100*gauss(gridX - 0.1, gridY, 0.1, 0.1)

nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)

gpsim1 = SsfmGPCUDA(psi0=psi,
                    gridX=gridX,
                    gridY=gridY,
                    m=m,
                    nR0=nR,
                    alpha=alpha,
                    Gamma=Gamma,
                    gammalp=gammalp,
                    R=R,
                    pump=pump1,
                    G=G,
                    eta=eta,
                    constV=constV,
                    dt=dt)

gpsim2 = SsfmGPCUDA(psi0=psi,
                    gridX=gridX,
                    gridY=gridY,
                    m=m,
                    nR0=nR,
                    alpha=alpha,
                    Gamma=Gamma,
                    gammalp=gammalp,
                    R=R,
                    pump=pump2,
                    G=G,
                    eta=eta,
                    constV=constV,
                    dt=dt)

nframes = 1024
fps = 24
fig, ax = plt.subplots()
im = ax.imshow(normSqr(gpsim2.psi - gpsim1.psi).real.cpu().detach().numpy(),
               origin='lower',
               extent=[startX, endX, startY, endY])
vmin = 0
vmax = 80
im.set_clim(vmin, vmax)
ax.set_xlabel(r'x ($\mu$m)')
ax.set_ylabel(r'y ($\mu$m)')
fig.colorbar(im, ax=ax)


def init():
    return [im]


def animate_heatmap(frame):
    gpsim1.step()
    gpsim2.step()
    data = normSqr(gpsim2.psi - gpsim1.psi).real.detach().cpu().numpy()
    im.set_data(data)
    return [im]


anim = animation.FuncAnimation(fig,
                               animate_heatmap,
                               init_func=init,
                               frames=nframes)
FFwriter = animation.FFMpegWriter(fps=fps,
                                  metadata={'copyright': 'Public Domain'})
anim.save('animations/chaosingp.mp4', writer=FFwriter)
