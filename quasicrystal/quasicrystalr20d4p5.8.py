import numpy as np
import torch
from src.solvers import SsfmGPCUDA
from src.penrose import makeSunGrid
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
dkx = 2 * kxmax / samplesX
dt = 0.1
m = 0.5
psi = 0.1*torch.rand((samplesY, samplesX), dtype=torch.cfloat)

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


radius = 20
divisions = 4
pumpStrength = 5.8
points = makeSunGrid(radius, divisions)
pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
pumpPos = np.zeros((samplesY, samplesX))
for p in points:
    addition = pumpStrength*gauss(gridX - p.real, gridY - p.imag, 0.1, 0.1)
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

nframes = 1536
fps = 24
fig, [ax1, ax2] = plt.subplots(1,2)
fig.dpi = 300
fig.figsize = (6.4, 3.6)
extentr = [startX*3/4, endX*3/4, startY*3/4, endY*3/4]
extentk = [-kxmax/2, kxmax/2, -kymax/2, kymax/2]
im1 = ax1.imshow(normSqr(gpsim.psi[63:512-64, 63:512-64]).real.cpu().detach().numpy(),
                 origin='lower',
                 extent=extentr)
im2 = ax2.imshow(normSqr(gpsim.psik[127:512-128, 127:512-128]).real.cpu().detach().numpy(),
                 origin='lower',
                 extent=extentk)
im1.set_clim(0, 0.03)

cmap0 = ListedColormap(['#00000000', '#ff6347ff'])
positions = ax1.scatter([p.real for p in points],
                        [p.imag for p in points],
                        s=0.5,
                        linewidths=0,
                        color='#ff6347')
ax1.set_xlabel(r'x ($\mu$m)')
ax1.set_ylabel(r'y ($\mu$m)')
plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
bleh = np.zeros((nframes, samplesX), dtype=complex)

for _ in range(1500):
    gpsim.step()

def init():
    return [im1, im2]


def animate_heatmap(frame):
    gpsim.step()
    rdata = gpsim.psi[63:512-64, 63:512-64].detach().cpu().numpy()
    kdata = fftshift(gpsim.psik.detach().cpu().numpy())
    bleh[frame, :] = kdata[256, :]
    im1.set_data(normSqr(rdata).real)
    im2.set_data(np.log(normSqr(kdata[127:512-128, 127:512-128]).real + 1))
    ax1.set_title(f'$|\\psi_r|^2$, t = {gpsim.t} ps')
    ax2.set_title('log$(|\\psi_k|^2 + 1)$')
    vmin = np.min(np.log(normSqr(kdata).real + 1))
    vmax = np.max(np.log(normSqr(kdata).real + 1))
    im2.set_clim(vmin, vmax)
    return [im1, im2]


anim = animation.FuncAnimation(fig,
                               animate_heatmap,
                               init_func=init,
                               frames=nframes,
                               blit=False)
FFwriter = animation.FFMpegWriter(fps=fps,
                                  metadata={'copyright': 'Public Domain'})
anim.save(f'animations/penroser{radius}d{divisions}p{pumpStrength}.mp4', writer=FFwriter)

Emax = np.pi / dt
bleh = fftshift(ifft(bleh, axis=0))
plt.cla()
fig, ax = plt.subplots()
im = ax.imshow(np.log(normSqr(bleh).real),
               origin='lower',
               extent=[-kxmax, kxmax, -Emax, Emax])
plt.savefig(f'graphs/penrosedispersionr{radius}d{divisions}p{pumpStrength}.pdf')
