import os
from argparse import ArgumentParser, BooleanOptionalAction

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cmaps
from matplotlib import rc

from src.solvers import npnormSqr

parser = ArgumentParser()
parser.add_argument("file")
parser.add_argument("--log", action=BooleanOptionalAction)
parser.add_argument("--psir")
parser.add_argument("--psik")
parser.add_argument("--psiksqrt", action=BooleanOptionalAction)
parser.add_argument("--normalise", action=BooleanOptionalAction)
args = parser.parse_args()

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
a = np.load(args.file)
if args.psir is not None:
    psir = npnormSqr(a["rpsidata"])
    # psir /= np.max(psir)
    extentr = a["extentr"]
    fig, ax = plt.subplots(layout="constrained")
    im = ax.imshow(
        psir,
        origin="lower",
        interpolation="none",
        extent=extentr,
        aspect="equal",
        cmap=cmaps["plasma"],
    )
    points = np.load("pumppositions.npy")
    # points = filterByRadius(points, 15.3) * (14 / (np.sqrt(3) / 2))
    ax.scatter(points[:, 0], points[:, 1], s=9, c="black", linewidths=0.0)
    ax.scatter(points[:, 0], points[:, 1], s=6, c="green", linewidths=0.0)
    zoom = 0.7
    ax.set_xlim(extentr[0] * zoom, extentr[1] * zoom)
    ax.set_ylim(extentr[2] * zoom, extentr[3] * zoom)
    ax.set_title(r"$|\psi_r|^2$, normalised")
    ax.set_xlabel(r"$x$ [µm$^{-1}$]")
    ax.set_ylabel(r"$y$ [µm]")

    # cb = plt.colorbar(im)
    fig.savefig(os.path.join(args.psir), bbox_inches="tight")

if args.psik is not None:
    psik = npnormSqr(a["kpsidata"])
    psik /= np.max(psik)
    extentk = [x / 2 for x in a["extentk"]]
    start = np.shape(psik)[0] // 4
    end = np.shape(psik)[0] - start
    fig, ax = plt.subplots()
    im = ax.imshow(
        psik[start:end, start:end],
        origin="lower",
        interpolation="none",
        extent=extentk,
        norm=colors.LogNorm(vmin=np.exp(-18), vmax=1.0),
    )
    ax.set_title(r"$|\psi_k|^2$")
    ax.set_xlabel(r"k_x [µm$^{-1}$]")
    ax.set_ylabel(r"y [µm$^{-1}$]")

    cb = plt.colorbar(im)
    fig.savefig(os.path.join(args.psik))
