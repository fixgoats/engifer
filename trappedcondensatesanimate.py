import os
import time
from pathlib import Path
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as tfft
from matplotlib import animation, cm

from src.solvers import hbar, newRunSimAnimate, npnormSqr, runSimPlain

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

now = gmtime()
day = strftime("%Y-%m-%d", now)
timeofday = strftime("%H.%M", now)
basedir = os.path.join("data/trap", day, timeofday)
Path(basedir).mkdir(parents=True, exist_ok=True)

gammalp = 0.2
constV = -0.5j * gammalp
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 26.4
Gamma = 0.1
eta = 2
initTime = 8
recordingTime = 10
fps = 24
nds = 21
prerun = 4000
nFrames = 240
framesPerD = nFrames // nds
nElementsX = 256
dt = 0.1
hbar = 6.582119569e-1  # meV * ps
m = 0.32
startX = -48
endX = 48
dx = (endX - startX) / nElementsX

kmax = np.pi / dx
dk = 2 * kmax / nElementsX
L0 = 1.1
r0 = 3.9
# seed = 2001124
seed = None
if seed is not None:
    gen = torch.random.manual_seed(seed=seed)
else:
    gen = torch.Generator()


def pumpprofile(x, y, L, r, beta):
    return (L * L) ** 2 / ((x**2 + beta * y**2 - r**2) ** 2 + (L * L) ** 2)


beta = 1.0
t1 = time.time()
nR = torch.zeros((nElementsX, nElementsX), device="cuda", dtype=torch.float)
k = torch.arange(-kmax, kmax, dk, device="cuda").type(dtype=torch.cfloat)
k = tfft.fftshift(k)
kxv, kyv = torch.meshgrid(k, k, indexing="xy")
kTimeEvo = torch.exp(-0.5j * hbar * (dt / m) * (kxv * kxv + kyv * kyv))
x = torch.arange(startX, endX, dx)
xv, yv = torch.meshgrid(x, x, indexing="xy")
xv = xv.type(dtype=torch.cfloat).to(device="cuda")
yv = yv.type(dtype=torch.cfloat).to(device="cuda")
psi = 2 * torch.rand((nElementsX, nElementsX), generator=gen, dtype=torch.cfloat).to(
    device="cuda"
) - (1 + 1j)

excitonFeed = torch.zeros(
    (nElementsX, nElementsX, nFrames), dtype=torch.float, device="cuda"
)
psiFeed = torch.zeros(
    (nElementsX, nElementsX, nFrames),
    dtype=torch.cfloat,
    device="cuda",
)

pump = (
    pumpStrength
    * (
        pumpprofile(xv - 5.1, yv, L0, r0, beta)
        + pumpprofile(xv + 5.1, yv, L0, r0, beta)
    ).real
)
constpart = constV + (G * eta / Gamma) * pump
psi, nR = runSimPlain(
    psi,
    nR,
    kTimeEvo,
    constpart,
    pump,
    dt,
    alpha,
    G,
    R,
    Gamma,
    prerun,
)

ds = np.linspace(5.1, 6, nds)
for i, d in enumerate(ds):
    pump = (
        pumpStrength
        * (
            pumpprofile(xv - d, yv, L0, r0, beta)
            + pumpprofile(xv + d, yv, L0, r0, beta)
        ).real
    )
    constpart = constV + (G * eta / Gamma) * pump

    psi, nR = newRunSimAnimate(
        psi,
        nR,
        kTimeEvo,
        constpart,
        pump,
        dt,
        alpha,
        G,
        R,
        Gamma,
        0,
        framesPerD,
        12,
        psiFeed[:, :, i * framesPerD :],
        excitonFeed[:, :, i * framesPerD :],
    )

animationFeed = psiFeed.detach().cpu().numpy()
filter_array = npnormSqr(animationFeed) > 1e-7
animationFeed *= filter_array
np.savez_compressed("animationarray", animationFeed)
fig, ax = plt.subplots()
fig.set_dpi(150)
fig.set_figwidth(8)
fig.set_figheight(8)

im = ax.imshow(
    npnormSqr(animationFeed[:, :, 0]),
    interpolation="none",
    origin="lower",
    extent=[startX, endX, startX, endX],
    aspect="equal",
)
cb = plt.colorbar(im)


def init1():
    return [im]


def animate_heatmap1(frame):
    data = npnormSqr(animationFeed[:, :, frame])
    im.set_data(data)
    im.set_clim(np.min(data), np.max(data))
    return [im]


anim = animation.FuncAnimation(
    fig,
    animate_heatmap1,
    init_func=init1,
    frames=nFrames,
    blit=True,
)
FFwriter = animation.FFMpegWriter(fps=fps, metadata={"copyright": "Public Domain"})

anim.save(f"coarsepumpscan.mp4", writer=FFwriter)

fig, ax = plt.subplots()
fig.set_dpi(150)
fig.set_figwidth(8)
fig.set_figheight(8)

im = ax.imshow(
    np.angle(animationFeed[:, :, 0]),
    interpolation="none",
    origin="lower",
    extent=[startX, endX, startX, endX],
    aspect="equal",
)
cb = plt.colorbar(im)


def init():
    return [im]


def animate_heatmap(frame):
    data = np.angle(animationFeed[:, :, frame])
    im.set_data(data)
    # im.set_clim(np.min(data), np.max(data))
    return [im]


anim = animation.FuncAnimation(
    fig,
    animate_heatmap,
    init_func=init,
    frames=nFrames,
    blit=True,
)
FFwriter = animation.FFMpegWriter(fps=fps, metadata={"copyright": "Public Domain"})

anim.save(f"lesssep.mp4", writer=FFwriter)
t2 = time.time()
print(f"Finished in {t2 - t1} seconds")
