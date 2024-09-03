import os
from argparse import ArgumentParser

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cmaps
from matplotlib import rc

from src.solvers import npnormSqr

parser = ArgumentParser()
parser.add_argument("files", type=str, nargs=3)
parser.add_argument("--out")
args = parser.parse_args()

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
fig = plt.figure(figsize=(8.4, 6.0))
subfigs = fig.subfigures(2, 1, hspace=0.0)
axsTop = subfigs[0].subplots(1, 3, gridspec_kw={"wspace": 0.0})
for ax in axsTop:
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axsTop[0].set_ylabel(r"$y$ [µm]")
# subfigs[0].suptitle(r"$|\psi_r|^2$")
axsBot = subfigs[1].subplots(1, 3, gridspec_kw={"wspace": 0.0})
# subfigs[1].suptitle(r"$|\psi_k|^2$")
axsBot[0].set_ylabel(r"$k_y$ [µm$^{-1}$]")
# fig.set_size_inches(w=8.8, h=6)
for i, name in enumerate(args.files):
    a = np.load(name)
    extentr = a["extentr"]
    _ = axsTop[i].imshow(
        npnormSqr(a["rpsidata"]),
        origin="lower",
        interpolation="none",
        extent=extentr,
        aspect="equal",
        cmap=cmaps["plasma"],
    )
    zoomr = 0.7
    axsTop[i].set_xlim(extentr[0] * zoomr, extentr[1] * zoomr)
    axsTop[i].set_ylim(extentr[2] * zoomr, extentr[3] * zoomr)
    # axsTop[i].set_xlabel(r"$x$ [µm$^{-1}$]")
    psik = npnormSqr(a["kpsidata"])
    psik /= np.max(psik)
    extentk = a["extentk"]
    _ = axsBot[i].imshow(
        psik,
        origin="lower",
        interpolation="none",
        extent=extentk,
        aspect="equal",
        cmap=cmaps["plasma"],
        norm=colors.LogNorm(vmin=np.exp(-18), vmax=1),
    )
    zoomk = 0.5
    axsBot[i].set_xlim(extentk[0] * zoomk, extentk[1] * zoomk)
    axsBot[i].set_ylim(extentk[2] * zoomk, extentk[3] * zoomk)
    # axsBot[i].set_xlabel(r"$k_x$ [µm$^{-1}$]")
    if i > 0:
        axsTop[i].set_yticks([])
        axsBot[i].set_yticks([])

fig.savefig(os.path.join(args.out))
