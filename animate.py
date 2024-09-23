import os
import time
from argparse import ArgumentParser
from pathlib import Path
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm

from src.solvers import hbar, newRunSimAnimate, npnormSqr

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

parser = ArgumentParser()
parser.add_argument("file")
parser.add_argument("out")
args = parser.parse_args()

fig, ax = plt.subplots(1, 2)
fig.set_dpi(150)
fig.set_figwidth(16)
fig.set_figheight(8)

system = np.load(args.file)
psiFeed = system["psiFeed"]
extentr = system["extentr"]
dX = (extentr[1] - extentr[0]) / np.shape(psiFeed)[0]
dY = (extentr[3] - extentr[2]) / np.shape(psiFeed)[1]

im0 = ax[0].imshow(
    npnormSqr(psiFeed[:, :, 0]),
    interpolation="none",
    origin="lower",
    extent=extentr,
    aspect="equal",
)
im1 = ax[1].imshow(
    np.angle(psiFeed[:, :, 0]),
    interpolation="none",
    origin="lower",
    extent=extentr,
    aspect="equal",
)
im1.set_clim(-np.pi, np.pi)


def probCurrentAtPoint(psi, i, j):
    dPsiXLeft = psi[i, j] - psi[i - 1, j]
    dPsiXRight = psi[i + 1, j] - psi[i, j]
    dPsiYDown = psi[i, j] - psi[i, j - 1]
    dPsiYUp = psi[i, j + 1] - psi[i, j]
    dPsiX = 0.5 * (dPsiXLeft + dPsiXRight) / dX
    dPsiY = 0.5 * (dPsiYDown + dPsiYUp) / dY
    return np.array(
        [
            (psi[i, j].conj() * dPsiX).imag,
            (psi[i, j].conj() * dPsiY).imag,
        ]
    )


def probComponent(psi, i):
    psiGrad = np.gradient(psi)
    return -1j * (psi.conj() * psiGrad[i] - psi * psiGrad[i].conj())


xSamples = 16
ySamples = 16
imax = np.shape(psiFeed)[0] - 10
di = (imax - 10) // 16
indices = np.arange(10, imax, di)
print(indices)
xs = indices * dX + extentr[0]
print(xs)
xv, yv = np.meshgrid(xs, xs)
u = np.array(
    [
        npnormSqr(probCurrentAtPoint(psiFeed[:, :, 0], i, j)[0])
        for i in indices
        for j in indices
    ]
)
v = np.array(
    [
        npnormSqr(probCurrentAtPoint(psiFeed[:, :, 0], i, j)[1])
        for i in indices
        for j in indices
    ]
)
normFactor = np.max(u * u + v * v)

q = ax[1].quiver(xv, yv, u, v, scale=1.0)


def init():
    return [im0, im1, q]


def animateImages(frame):
    data = psiFeed[:, :, frame]
    intensity = npnormSqr(data)
    phase = np.angle(data)
    im0.set_data(intensity)
    im0.set_clim(np.min(intensity), np.max(intensity))
    im1.set_data(phase)
    u = np.array(
        [
            probCurrentAtPoint(psiFeed[:, :, frame], i, j)[0]
            for i in indices
            for j in indices
        ]
    )
    v = np.array(
        [
            probCurrentAtPoint(psiFeed[:, :, frame], i, j)[1]
            for i in indices
            for j in indices
        ]
    )
    q.set_UVC(u, v)

    return [im0, im1, q]


anim = animation.FuncAnimation(
    fig, animateImages, init_func=init, frames=np.shape(psiFeed)[2], blit=True
)
FFwriter = animation.FFMpegWriter(fps=24, metadata={"copyright": "Public Domain"})
anim.save(args.out, writer=FFwriter)
