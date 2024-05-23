import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--set")
args = parser.parse_args()

fig, ax = plt.subplots(nrows=2, ncols=4, gridspec_kw={"wspace": 0.00, "hspace": 0.15})
fig.set_size_inches(w=11, h=6)
fig.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.92)
cbar_ax = fig.add_axes([0.93, 0.05, 0.02, 0.9])

data = {}
sets = ["13.5", "11.1", "10.1", "7.8"]
for s in sets:
    data[s] = np.load(
        os.path.join("graphs", "fig5repro", args.set, f"psidata{s}.npy"),
        allow_pickle=True,
    ).item()

ims = {}
for i, s in enumerate(sets):
    tmpr = data[s]["rpsidata"]
    tmpr /= np.max(tmpr)
    startr = np.shape(tmpr)[0] // 6
    endr = np.shape(tmpr)[0] - startr
    tmpk = data[s]["kpsidata"]
    tmpk /= np.max(tmpk)
    startk = int(np.shape(tmpk)[0] / 2.5)
    endk = np.shape(tmpr)[0] - startk
    extentr = [2 * x / 3 for x in data[s]["extentr"]]
    extentk = [x / 5 for x in data[s]["extentk"]]
    ims[s + "r"] = ax[0, i].imshow(
        tmpr[startr:endr, startr:endr],
        aspect="equal",
        origin="lower",
        interpolation="none",
        extent=extentr,
    )
    ax[0, i].set_title(f"$|\\psi_r|^2$, D={s}")
    ims[s + "k"] = ax[1, i].imshow(
        tmpk[startk:endk, startk:endk],
        aspect="equal",
        origin="lower",
        interpolation="none",
        extent=extentk,
    )
    ax[1, i].set_title(f"$|\\psi_k|^2$, D={s}")
    if i > 0:
        ax[0, i].set_yticks([])
        ax[1, i].set_yticks([])


fig.colorbar(ims["13.5r"], cax=cbar_ax)

fig.savefig(os.path.join("graphs", "fig5repro", "fig5repro.pdf"))
