import numpy as np
import torch
from src.solvers import SsfmGPCUDA
import random
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

# constV = ((gridX / 50)**2 + (gridY / 50)**2)**8 - 0.5j*pars["gammalp"]
constV = -0.5j*pars["gammalp"]*torch.ones((samplesY, samplesX))


def normSqr(x):
    return x.conj() * x


points = torch.zeros((100, 2))
pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
i = 0
while i < 100:
    p = torch.tensor([random.uniform(4*startX/5, 4*endX/5),
                      random.uniform(4*startY/5, 4*endY/5)])
    if any(torch.sqrt(torch.sum((points-p)*(points-p), axis=1)) < 2*pars["sigma"]):
        continue
    points[i, :] = p
    i += 1

print('done making grid')

for p in points:
    pump += pars["pumpStrength"]*gauss(gridX - p[0],
                                       gridY - p[1],
                                       pars["sigma"],
                                       pars["sigma"])

nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
gpsim = SsfmGPCUDA(psi0=psi,
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
fps = 24
fig, [ax1, ax2] = plt.subplots(1, 2)
fig.dpi = 300
fig.figsize = (6.4, 3.6)
extentr = [startX, endX, startY, endY]
extentk = [-kxmax/2, kxmax/2, -kymax/2, kymax/2]
im1 = ax1.imshow(normSqr(gpsim.psi).real.cpu().detach().numpy(),
                 origin='lower',
                 extent=extentr)
ax1.set_title(r'$|\psi_r|^2, t = 0 ps$')
psik0 = normSqr(gpsim.psik).real.cpu().detach().numpy()
im2 = ax2.imshow(np.log(fftshift(psik0) + 1)[255:samplesY-256, 255:samplesX-256],
                 origin='lower',
                 extent=extentk)
ax2.set_title(r'$\ln(|\psi_k|^2 + 1)$')

positions = ax1.scatter(points[:, 0],
                        points[:, 1],
                        s=0.5,
                        linewidths=0.1,
                        color='#ff6347')
ax1.set_xlabel(r'x ($\mu$m)')
ax1.set_ylabel(r'y ($\mu$m)')
# fraction=0.046 and pad=0.04 are magic settings that just work for some reason
# to make the colorbar the same height as the graph.
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
bleh = np.zeros((nframes, samplesX), dtype=complex)

for _ in range(512):
    gpsim.step()


def init():
    return [im1, im2]


def animate_heatmap(frame):
    gpsim.step()
    rdata = normSqr(gpsim.psi).real.detach().cpu().numpy()
    kdata = gpsim.psik.detach().cpu().numpy()
    bleh[frame, :] = kdata[samplesY//2 - 1, :]
    kdata = np.log(normSqr(fftshift(kdata))[255:samplesY-256, 255:samplesX-255].real + 0.1)
    im1.set_data(rdata)
    im2.set_data(kdata)
    vmin = np.min(kdata)
    vmax = np.max(kdata)
    im2.set_clim(vmin, vmax)
    vmin = np.min(rdata)
    vmax = np.max(rdata)
    im1.set_clim(vmin, vmax)
    ax1.set_title(f'$|\\psi_r|^2$, t = {gpsim.t:.2f}')
    return [im1, im2]


anim = animation.FuncAnimation(fig,
                               animate_heatmap,
                               init_func=init,
                               frames=nframes,
                               blit=True)
FFwriter = animation.FFMpegWriter(fps=fps, metadata=pars)

path = f'animations/random'\
       f'p{pars["pumpStrength"]}'\
       f'n{pars["samplesX"]}'\
       f's{pars["sigma"]}.mp4'

anim.save(path, writer=FFwriter)

plt.cla()
fig, ax = plt.subplots()
rdata = normSqr(gpsim.psi).real.detach().cpu().numpy()
im = ax.imshow(rdata, origin='lower',
               extent=extentr)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
path = f'graphs/randomlastframer'\
       f'p{pars["pumpStrength"]}'\
       f'n{pars["samplesX"]}'\
       f's{pars["sigma"]}.pdf'
ax.set_title(r'$|\psi_r|^2$')
ax.set_xlabel(r'x ($\mu$m)')
ax.set_ylabel(r'y ($\mu$m)')
plt.savefig(path)

plt.cla()
fig, ax = plt.subplots()
kdata = normSqr(gpsim.psik).real.detach().cpu().numpy()
kdata = fftshift(kdata)[255:samplesY-256, 255:samplesX-256]
im = ax.imshow(np.log(kdata+1), origin='lower',
               extent=extentk)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
path = f'graphs/randomlastframek'\
       f'p{pars["pumpStrength"]}'\
       f'n{pars["samplesX"]}'\
       f's{pars["sigma"]}.pdf'
ax.set_title(r'$\ln(|\psi_k|^2 + 1)$')
ax.set_xlabel(r'k_x ($\hbar/\mu$ m)')
ax.set_ylabel(r'k_y ($\hbar/\mu$ m)')
plt.savefig(path)

Emax = np.pi / pars['dt']
bleh = fftshift(ifft(bleh, axis=0))
plt.cla()
fig, ax = plt.subplots()
im = ax.imshow(np.log(normSqr(bleh).real),
               origin='lower',
               aspect='auto',
               extent=[-kxmax, kxmax, -Emax, Emax])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
path = f'graphs/randomdispersion'\
       f'p{pars["pumpStrength"]}'\
       f'n{pars["samplesX"]}'\
       f's{pars["sigma"]}.pdf'
ax.set_title('Dispersion relation, logarithmic')
ax.set_xlabel(r'k_x ($\hbar/\mu$ m)')
ax.set_ylabel(r'E ($\hbar/ps)')
plt.savefig(path)
