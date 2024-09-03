from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from src.solvers import hbar, npnormSqr

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

parser = ArgumentParser()
parser.add_argument("--files", type=str, nargs="+")
parser.add_argument("--out")
args = parser.parse_args()

fig, ax = plt.subplots(layout="constrained")
names = ["Penrose", "Square", "Monotile", "Amorphous"]
ommax = hbar * np.pi / 0.2
for i, name in enumerate(args.files):
    spectrum = npnormSqr(np.load(name))
    spectrum /= np.max(spectrum)
    ax.plot([ommax * x / 512 for x in range(len(spectrum))], spectrum, label=names[i])

ax.set_xlim(0, ommax / 10)
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_ylabel(r"$\rho$")
ax.set_xlabel("E [meV]")
ax.legend()
fig.savefig(args.out, bbox_inches="tight")
