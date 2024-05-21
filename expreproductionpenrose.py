import argparse
import os
import shutil
import time
import tomllib
from pathlib import Path
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.penrose import filterByRadius, goldenRatio, makeSunGrid
from src.solvers import SsfmGPGPU, gauss, smoothnoise, tgauss

t1 = time.time()
now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
basedir = os.path.join("graphs", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--use-cached", required=False)
parser.add_argument("--check-penrose", required=False, action='store_true', default=None)
args = parser.parse_args()
if args.config is None and args.use_cached is None:
    exit("Need to specify config")

with open(f"{args.config}", "rb") as f:
    pars = tomllib.load(f)

shutil.copy(args.config, basedir)

cuda = torch.device('cuda')
endX = pars["endX"]
startX = pars["startX"]
samplesX = pars["samplesX"]
endY = pars["endY"]
startY = pars["startY"]
samplesY = pars["samplesY"]
dt = pars["dt"]
nframes = pars["nframes"]
dx = (endX - startX) / samplesX
dy = (endY - startY) / samplesY
x = np.arange(startX, endX, dx)
y = np.arange(startY, endY, dy)
gridX, gridY = np.meshgrid(y, x)
kxmax = np.pi / dx
kymax = np.pi / dy
dkx = 2 * kxmax / samplesX
psi = torch.from_numpy(smoothnoise(gridX, gridY))
radius = pars['radius']
divisions = pars['divisions']
points = filterByRadius(makeSunGrid(radius, divisions), pars['cutoff'])
pumpStrength = pars['pumpStrength']
sigma = pars['sigma']

if args.check_penrose is not None:
    pump = np.zeros((samplesY, samplesX))
    for p in points:
        pump += pumpStrength*gauss(gridX - p[0], gridY - p[1], sigma, sigma)
    fig, ax = plt.subplots()
    ax.imshow(pump)
    plt.show()
    exit()

gridX = torch.from_numpy(gridX).type(dtype=torch.cfloat).to(device='cuda')
gridY = torch.from_numpy(gridY).type(dtype=torch.cfloat).to(device='cuda')

# Coefficients for GP equation
alpha = pars['alpha']
gammalp = pars['gammalp']
Gamma = pars['Gamma']
G = pars['G']
R = pars['R']
eta = pars['eta']
constV = - 0.5j*gammalp * torch.ones((samplesX, samplesY), device=cuda)# - 1j*(torch.cosh((gridX/125)**2 + (gridY/125)**2) - 1)

minsep = pars["radius"] / (goldenRatio**pars["divisions"])
pars["minsep"] = minsep


pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat, device=cuda)
for p in points:
    pump += pumpStrength*tgauss(gridX - p[0], gridY - p[1], sigma, sigma)


nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
gpsim = SsfmGPGPU(
    dev=cuda,
    gridX=gridX,
    gridY=gridY,
    psi0=psi,
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
    dt=dt,
)

for _ in range(nframes):
    gpsim.step()

psi = gpsim.psi.detach().cpu().numpy()
psik = gpsim.psik.detach().cpu().numpy()
np.save(os.path.join(basedir, "psidata"), {'psir': psi, 'psik': psik})
t2 = time.time()
print(f"finished in {t2 - t1} seconds")
