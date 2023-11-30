import numpy as np
import torch
from src.solvers import SsfmGPGPU, npnormSqr, tnormSqr, hbar
from src.penrose import makeSunGrid, goldenRatio
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from matplotlib import animation
import tomllib
import argparse
from pathlib import Path
from datetime import date

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

datestamp = date.today()
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


if args.use_cached is False:
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

    pump = pars["pumpStrength"] * (gauss(gridX, gridY, pars["sigma"], pars["sigma"]))
    nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
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
                      dt=pars["dt"])

    nframes = 1024
    fps = 24
    fig, [ax1, ax2] = plt.subplots(1, 2)
    fig.dpi = 300
    fig.figsize = (6.4, 3.6)
    extentr = np.array([startX, endX, startY, endY])
    extentk = np.array([-kxmax/2, kxmax/2, -kymax/2, kymax/2])
    dispersion = np.zeros((nframes, samplesX), dtype=complex)

    for _ in range(pars["prerun"]):
        gpsim.step()

    for i in range(pars["nframes"]):
        gpsim.step()
        rdata = gpsim.psi.detach().cpu().numpy()
        dispersion[i, :] = rdata[samplesY//2 - 1, :]

    Path("tmp").mkdir(parents=True, exist_ok=True)
    rdata = gpsim.psi.detach().cpu().numpy()
    np.save("tmp/rdata.npy", rdata)
    kdata = gpsim.psik.detach().cpu().numpy()
    np.save("tmp/kdata.npy", kdata)
    np.save("tmp/disp.npy", dispersion)
    np.save("tmp/extentr.npy", extentr)
    np.save("tmp/extentk.npy", extentk)
    Emax = hbar * np.pi / pars['dt']
    extentE = [-kxmax, kxmax, 0, Emax]
    np.save("tmp/extentE.npy", extentE)


else:
    rdata = np.load("tmp/rdata.npy")
    kdata = np.load("tmp/kdata.npy")
    dispersion = np.load("tmp/disp.npy")
    extentr = np.load("tmp/extentr.npy")
    extentk = np.load("tmp/extentk.npy")
    extentE = np.load("tmp/extentE.npy")
    samplesY, samplesX = np.shape(rdata)

plt.cla()
fig, ax = plt.subplots()
rdata = rdata[samplesY//6-1:samplesY-samplesY//6,
              samplesX//6-1:samplesX-samplesX//6]
im = ax.imshow(npnormSqr(rdata), origin='lower',
               extent=2*extentr/3)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
basedir = f"graphs/{datestamp}"
Path(basedir).mkdir(parents=True, exist_ok=True)
name = 'examplerplot'
ax.set_title('$|\\psi_r|^2$, cropped view.')
ax.set_xlabel(r'x (µm)')
ax.set_ylabel(r'y (µm)')
ax.set_ylim(2*extentr[2]/3, 2*extentr[3]/3)
ax.set_xlim(2*extentr[0]/3, 2*extentr[1]/3)
plt.savefig(f'{basedir}/{name}.pdf')
plt.savefig(f'{basedir}/{name}.png')
print(f'Made r-space plot {name} in {basedir}')

startky = samplesY // 4 - 1
endky = samplesY - samplesY // 4
startkx = samplesX // 4 - 1
endkx = samplesX - samplesX // 4
kdata = npnormSqr(fftshift(kdata)[startky:endky, startkx:endky])
plt.cla()
fig, ax = plt.subplots()
im = ax.imshow(kdata, origin='lower',
               extent=extentk)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
name = 'examplekplot'
ax.set_title(r'$|\psi_k|^2$')
ax.set_xlabel(r'$k_x$ ($\mu m^{-1}$)')
ax.set_ylabel(r'$k_y$ ($\mu m^{-1}$)')
plt.savefig(f'{basedir}/{name}.pdf')
plt.savefig(f'{basedir}/{name}.png')
print(f'Made k-space plot {name} in {basedir}')

plt.cla()
fig, ax = plt.subplots()
im = ax.imshow(np.log(kdata+np.exp(-10)), origin='lower',
               extent=extentk)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
name = 'examplelogkplot'
ax.set_title(r'$\ln(|\psi_k|^2 + e^{-10})$')
ax.set_xlabel(r'$k_x$ ($\mu m^{-1}$)')
ax.set_ylabel(r'$k_y$ ($\mu m^{-1}$)')
plt.savefig(f'{basedir}/{name}.pdf')
plt.savefig(f'{basedir}/{name}.png')
print(f'Made logarithmic k-space plot {name} in {basedir}')

dispersion = fftshift(fft(ifft(dispersion, axis=0), axis=1))
start = dispersion.shape[0] // 2 - 1
dispersion = dispersion[start:, :]
plt.cla()
fig, ax = plt.subplots()
im = ax.imshow(np.sqrt(npnormSqr(dispersion)),
               aspect='auto',
               origin='lower',
               extent=extentE)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
name = 'exampledispersionplot'
ax.set_title(r'$\rho(E, k_x)$')
ax.set_xlabel(r'$k_x$ ($\mu m^{-1}$)')
ax.set_ylabel(r'$E$ (meV)')
plt.savefig(f'{basedir}/{name}.pdf')
plt.savefig(f'{basedir}/{name}.png')
print(f'Made dispersion relation plot {name} in {basedir}')

plt.cla()
fig, ax = plt.subplots()
im = ax.imshow(np.log(np.sqrt(npnormSqr(dispersion) + np.exp(-10))),
               aspect='auto',
               origin='lower',
               extent=extentE)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
name = 'examplelogdispersionplot'
ax.set_title(r'$\ln(\rho(E, k_x) + e^{-10})$')
ax.set_xlabel(r'$k_x$ ($\mu m^{-1}$)')
ax.set_ylabel(r'$E$ (meV)')
plt.savefig(f'{basedir}/{name}.pdf')
plt.savefig(f'{basedir}/{name}.png')
print(f'Made logarithmic dispersion relation plot {name} in {basedir}')
