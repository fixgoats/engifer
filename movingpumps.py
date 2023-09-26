import numpy as np
import torch
from torch.fft import fft, ifft, fft2, fftshift
from src.solvers import SsfmGPCUDA, findLocalMaxima
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output')
args = parser.parse_args()
if args.output is None:
    exit('Need to specify filename for graph')


def gauss(x, y):
    return torch.exp(-x * x - y * y)


def exactsol(x, y, t, a, m):
    return (a / (a + 1.0j * t / m)) \
        * torch.exp(-(x * x + y * y) / (2 * (a + 1.0j * t / m)))


cuda = torch.device('cuda')
samplesX = 256
samplesY = 256
startX = -20  # micrometers
endX = 20
startY = -20
endY = 20
dx = (endX - startX) / samplesX
dy = (endY - startY) / samplesY
x = torch.arange(startX, endX, dx)
x = x.type(dtype=torch.cfloat)
gridY, gridX = torch.meshgrid(x, x, indexing='ij')
kxmax = np.pi / dx
kymax = np.pi / dy
dt = 0.15
m = 0.5
psi = torch.rand((samplesY, samplesX), dtype=torch.cfloat)

# Coefficients for GP equation
alpha = 0.01
gammalp = 2
Gamma = 1
G = 0.1
R = 2
eta = 1
constV = ((gridX / 16)**2 + (gridY / 16)**2)**8 - 0.5j*gammalp


def normSqr(x):
    return x.conj() * x


pump = 100 * (gauss(gridX - 2, gridY) + gauss(gridX + 2, gridY))
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
npumps = 31
distances = np.zeros(npumps)
energies = np.zeros(npumps)
maxEnergy = np.pi / dt  # hbar / ps
dE = 2 * maxEnergy / nframes


uhh = []
separations = np.linspace(2, 3.24, npumps)
ds = []
wholepicture = np.zeros((nframes, npumps))
for j, d in enumerate(separations):
    dispersion = torch.zeros((nframes, samplesX), dtype=torch.cfloat, device=cuda)
    pump = 50 * (gauss((gridX - d), (gridY)) + gauss((gridX + d), (gridY)))
    gpsim.pump = pump.to(device=cuda)
    for i in range(nframes):
        gpsim.step()
        dispersion[i, :] = gpsim.psi[127, :]

    # dispersion = torch.flip(dispersion, [0])
    if j % 10 == 0:
        fig, ax = plt.subplots()
        tmp = normSqr(dispersion).real.cpu().detach().numpy()
        im = ax.imshow(tmp,
                       origin='lower',
                       aspect='auto',
                       extent=[startX, endX, 0, nframes*dt])
        fig.colorbar(im, ax=ax)
        plt.savefig(f'graphs/rsnapshot{j//10}.pdf')
    window = fftshift(ifft(fft(dispersion, dim=0), dim=1))
    window = (window.conj() * window).real
    if j % 10 == 0:
        fig, ax = plt.subplots()
        tmp = window.cpu().detach().numpy()
        im = ax.imshow(tmp,
                       origin='lower',
                       aspect='auto',
                       extent=[-kxmax, kxmax, -maxEnergy, maxEnergy])
        fig.colorbar(im, ax=ax)
        plt.savefig(f'graphs/ksnapshot{j//10}.pdf')
    window = torch.sum(window, 1) / (samplesX*samplesY)
    bleh = findLocalMaxima(window, fuzz=2e-3)
    uhh.append(bleh)
    for val in bleh:
        ds.append(d)
    wholepicture[:, j] = window.cpu().detach().numpy()


# x[0] is the index in window where the peak was found. If the first element in
# window corresponds to the highest energy, then maxEnergy - dE * x[0] should
# be the corresponding energy value.
energies = [maxEnergy - dE * x[0] for bleh in uhh for x in bleh]
fig, ax = plt.subplots()
ax.set_ylabel(r'$E$ ($\hbar$ / ps)')
ax.set_xlabel(r'$d$ ($\mu$ m)')
plt.plot(ds, energies, 'r+')
plt.savefig(args.output)

fig, ax = plt.subplots()
im = ax.imshow(wholepicture,
               aspect='auto',
               extent=[2, 5, -maxEnergy, maxEnergy])
ax.set_ylabel(r'E ($\hbar$/ps)')
ax.set_xlabel(r'd ($\mu$m)')
fig.colorbar(im, ax=ax)
plt.savefig('graphs/nonflippedwholepicture.pdf')
