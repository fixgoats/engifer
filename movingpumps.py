import numpy as np
import torch
from torch.fft import fft2, fftshift
from src.solvers import SsfmGPCUDA
import matplotlib.pyplot as plt


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
dt = 0.02
m = 0.5
psi = torch.rand((samplesY, samplesX), dtype=torch.cfloat)
psik0 = fftshift(fft2(psi, norm='ortho'))
psik0 = psik0[192:320, 192:320]
fps = 24

# Coefficients for GP equation
alpha = 0.01
gammalp = 2
Gamma = 1
G = 0.1
R = 2
eta = 1

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
                   dt=dt)

nframes = 512
npumps = 100
distances = np.zeros(npumps)
energies = np.zeros(npumps)
def findLocalMaxima(vector, fuzz):
    idxValPairs = []
    last = vector[0]
    ascending = True
    for i, val in enumerate(vector):
        if ascending and abs(val - last) > fuzz and val < last:
            idxValPairs.append((i-1, last.item()))
            ascending = False
        elif not ascending and abs(val - last) > fuzz and val >= last:
            ascending = True

        last = val
    idxValPairs = sorted(idxValPairs, key=lambda x: x[1])
    return idxValPairs

uhh = []
separations = np.linspace(2, 7, npumps)
ds = []
for d in separations:
    dispersion = torch.zeros((nframes, samplesX), dtype=torch.cfloat, device=cuda)
    pump = 100 * (gauss(gridX - d, gridY)\
            + gauss(gridX + d, gridY))
    gpsim.pump = pump.to(device=cuda)
    for i in range(nframes):
        gpsim.step()
        dispersion[i, :] = gpsim.psi[255, :]

    dispersion = torch.flip(dispersion, [0])
    window = fftshift(fft2(dispersion, norm='ortho'))
    window = (window.conj() * window).real
    window = torch.sum(window, 1) / samplesX
    bleh = findLocalMaxima(window[270:290], fuzz=2e-6)
    uhh.append(bleh)
    for val in bleh:
        ds.append(d)

maxEnergy = np.pi / dt # hbar / ps
dE = 2 * maxEnergy / nframes

energies = [maxEnergy - dE * (242 + x[0]) for bleh in uhh for x in bleh]
fig, ax = plt.subplots()
ax.set_ylabel(r'$E$ ($\hbar$ / ps)')
ax.set_xlabel(r'$d$ ($\mu$ m)')
plt.plot(ds, energies, 'ro')
plt.savefig("graphs/energies.pdf")
