import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

parser = ArgumentParser()
parser.add_argument("--set")
args = parser.parse_args()

fig, ax = plt.subplots(nrows=2, ncols=3, gridspec_kw={"wspace": 0.00, "hspace": 0.18})
fig.set_size_inches(w=8.8, h=6)
fig.subplots_adjust(left=0.06, bottom=0.05, top=0.95, right=0.90)
cbar_ax1 = fig.add_axes([0.92, 0.537, 0.02, 0.412])
cbar_ax2 = fig.add_axes([0.92, 0.05, 0.02, 0.412])

data = {}
defects = [0, 5, 21]
for n in defects:
    data[n] = np.load(
        os.path.join("graphs", "fig4repro", args.set, f"psidata{n}.npy"),
        allow_pickle=True,
    ).item()

ims = {0: {}, 5: {}, 21: {}}
for i, n in enumerate(defects):
    tmpr = data[n]["rpsidata"]
    tmpr /= np.max(tmpr)
    startr = np.shape(tmpr)[0] // 6
    endr = np.shape(tmpr)[0] - startr
    tmpk = data[n]["kpsidata"]
    tmpk /= np.max(tmpk)
    startk = int(np.shape(tmpk)[0] / 2.5)
    endk = np.shape(tmpr)[0] - startk
    extentr = [2 * x / 3 for x in data[n]["extentr"]]
    extentk = [x / 5 for x in data[n]["extentk"]]
    ims[n]["r"] = ax[0, i].imshow(
        tmpr[startr:endr, startr:endr],
        aspect="equal",
        origin="lower",
        interpolation="none",
        extent=extentr,
    )
    ax[0, i].set_title(f"$|\\psi_r|^2$, n={n}")
    ims[n]["k"] = ax[1, i].imshow(
        tmpk[startk:endk, startk:endk],
        aspect="equal",
        origin="lower",
        interpolation="none",
        extent=extentk,
        norm=LogNorm(vmin=np.exp(-20), vmax=np.max(tmpk)),
    )
    ax[1, i].set_title(f"$|\\psi_k|^2$, n={n}")
    if i > 0:
        ax[0, i].set_yticks([])
        ax[1, i].set_yticks([])


fig.colorbar(ims[0]["r"], cax=cbar_ax1)
fig.colorbar(ims[0]["k"], cax=cbar_ax2)

fig.savefig(os.path.join("graphs", "fig4repro", "fig4repro.pdf"))
