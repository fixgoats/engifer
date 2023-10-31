import numpy as np
import torch
from src.solvers import SsfmGPCUDA, npnormSqr, tnormSqr, hbar
from src.monotile import makegrid
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
endX = pars["endX"] # lengths in micrometers
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
psi = 0.1 * torch.rand((samplesY, samplesX), dtype=torch.cfloat)

# constV = ((gridX / 50)**2 + (gridY / 50)**2)**8 - 0.5j*pars["gammalp"]
constV = -0.5j*pars["gammalp"]*torch.ones((samplesY, samplesX))

points = makegrid(pars["scale"])
pars["minsep"] = pars["scale"] * 0.5
pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
for p in points:
    pump += pars["pumpStrength"]*gauss(gridX - p[0],
                                       gridY - p[1] - 20,
                                       pars["sigma"],
                                       pars["sigma"])

nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
gpsim = SsfmGPCUDA(dev=cuda,
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
                   dt=pars["dt"])

nframes = 1024
fps = 24
fig, [ax1, ax2] = plt.subplots(1, 2)
fig.dpi = 300
fig.figsize = (6.4, 3.6)
extentr = [startX, endX, startY, endY]
extentk = [-kxmax/2, kxmax/2, -kymax/2, kymax/2]
rdata = gpsim.psi.cpu().detach().numpy()
im1 = ax1.imshow(npnormSqr(rdata),
                 origin='lower',
                 extent=extentr)
ax1.set_title(r'$|\psi_r|^2, t = 0 ps$')
psik0 = tnormSqr(gpsim.psik).real.cpu().detach().numpy()
im2 = ax2.imshow(np.log(fftshift(psik0) + 1)[255:samplesY-256, 255:samplesX-256],
                 origin='lower',
                 extent=extentk)
ax2.set_title(r'$\ln(|\psi_k|^2 + 1)$')

positions = ax1.scatter(points[:, 0],
                        points[:, 1]+20,
                        s=0.5,
                        linewidths=0.1,
                        color='#ff6347')
ax1.set_xlabel(r'x ($\mu$m)')
ax1.set_ylabel(r'y ($\mu$m)')
# fraction=0.046 and pad=0.04 are magic settings that just work for some reason
# to make the colorbar the same height as the graph.
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
bleh = np.zeros((2*nframes, samplesX), dtype=complex)

for i in range(2*nframes):
    gpsim.step()
    rdata = gpsim.psi.detach().cpu().numpy()
    bleh[i, :] = rdata[samplesY//2 - 1, :]


# def init():
#     return [im1, im2]
# 
# 
# def animate_heatmap(frame):
#     gpsim.step()
#     rdata = gpsim.psi.detach().cpu().numpy()
#     kdata = gpsim.psik.detach().cpu().numpy()
#     bleh[frame, :] = rdata[samplesY//2 - 1, :]
#     kdata = np.log(npnormSqr(fftshift(kdata))[255:samplesY-256, 255:samplesX-255].real + 1)
#     im1.set_data(npnormSqr(rdata))
#     im2.set_data(kdata)
#     vmin = np.min(kdata)
#     vmax = np.max(kdata)
#     im2.set_clim(vmin, vmax)
#     vmin = np.min(npnormSqr(rdata))
#     vmax = np.max(npnormSqr(rdata))
#     im1.set_clim(vmin, vmax)
#     ax1.set_title(f'$|\\psi_r|^2$, t = {gpsim.t:.2f}')
#     return [im1, im2]
# 
# 
# anim = animation.FuncAnimation(fig,
#                                animate_heatmap,
#                                init_func=init,
#                                frames=nframes,
#                                blit=True)
# FFwriter = animation.FFMpegWriter(fps=fps, metadata=pars)
# 
# path = f'animations/monotiler{pars["scale"]}'\
#        f'p{pars["pumpStrength"]}'\
#        f'n{pars["samplesX"]}'\
#        f's{pars["sigma"]}.mp4'
# 
# anim.save(path, writer=FFwriter)

logscale = ""
if pars["log"]:
    logscale = "logscale"

plt.cla()
fig, ax = plt.subplots()
fig.dpi = 300
fig.figsize = (6.4, 3.6)
rdata = gpsim.psi.detach().cpu().numpy()
im = ax.imshow(npnormSqr(rdata), origin='lower',
               extent=extentr)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
path = f'graphs/monotilelastframerr{pars["scale"]}'\
       f'p{pars["pumpStrength"]}'\
       f'n{pars["samplesX"]}'\
       f's{pars["sigma"]}.pdf'
print(f"made r-space graph {path}")
ax.set_title(r'$|\psi_r|^2$')
ax.set_xlabel(r'$x$ ($\mu$m)')
ax.set_ylabel(r'$y$ ($\mu$m)')
plt.savefig(path)

plt.cla()
fig, ax = plt.subplots()
fig.dpi = 300
fig.figsize = (6.4, 3.6)
kdata = gpsim.psik.detach().cpu().numpy()
kdata = fftshift(npnormSqr(kdata))[255:samplesY-256, 255:samplesX-256]
if pars["log"]:
    kdata = np.log(kdata + np.exp(-10))

im = ax.imshow(kdata, origin='lower',
               extent=extentk)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
path = f'graphs/monotilelastframekr{pars["scale"]}'\
       f'p{pars["pumpStrength"]}'\
       f'n{pars["samplesX"]}'\
       f's{pars["sigma"]}{logscale}.pdf'
print(f"made k-space graph {path}")
if pars["log"]:
    ax.set_title(r'$\ln(|\psi_k|^2 + e^{-10})$')
else:
    ax.set_title(r'$|\psi_k|^2$')
ax.set_xlabel(r'$k_x$ ($\mu m^{-1}$)')
ax.set_ylabel(r'$k_y$ ($\mu m^{-1}$)')
plt.savefig(path)

Emax = hbar * np.pi / pars['dt']
bleh = fftshift(fft(ifft(bleh, axis=0), axis=1))
bleh = np.sqrt(npnormSqr(bleh))
plt.cla()
fig, ax = plt.subplots()
fig.dpi = 300
fig.figsize = (6.4, 3.6)
if pars["log"]:
    bleh = np.log(bleh + np.exp(-10))

    im = ax.imshow(bleh[nframes:2*nframes-1, :],
                   origin='lower',
                   aspect='auto',
                   extent=[-kxmax, kxmax, 0, Emax])
    vmin = np.min(bleh)
    vmax = np.max(bleh)
    im.set_clim(vmin, vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = f'graphs/monotiledispersionr{pars["scale"]}'\
           f'p{pars["pumpStrength"]}'\
           f'n{pars["samplesX"]}'\
           f's{pars["sigma"]}{logscale}.pdf'
    print(f"made dispersion graph {path}")
    ax.set_title(f'$E(k_x)$ {logscale}')
    ax.set_xlabel(r'$k_x$ ($\mu m^{-1}$)')
    ax.set_ylabel(r'$E$ (meV)')
    plt.savefig(path)
else:
    im = ax.imshow(bleh,
                   origin='lower',
                   aspect='auto',
                   extent=[-kxmax, kxmax, 0, Emax])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = f'graphs/monotiledispersionr{pars["scale"]}'\
           f'p{pars["pumpStrength"]}'\
           f'n{pars["samplesX"]}'\
           f's{pars["sigma"]}{logscale}.pdf'
    print(f"made dispersion graph {path}")
    ax.set_title(f'$E(k_x)$ {logscale}')
    ax.set_xlabel(r'$k_x$ ($\mu m^{-1}$)')
    ax.set_ylabel(r'$E$ (meV)')
    plt.savefig(path)
