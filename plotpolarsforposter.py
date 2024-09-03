from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from src.penrose import goldenRatio

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

parser = ArgumentParser()
parser.add_argument("--files", type=str, nargs="+")
parser.add_argument("--out")
args = parser.parse_args()

radius = 14 * goldenRatio**6
width = 2 * (radius + 30)
pixelarea = (width / 1024) ** 2
fig, ax = plt.subplots(layout="constrained")
names = ["Penrose", "Square", "Monotile", "Amorphous"]
for i, name in enumerate(args.files):
    npolars = np.load(name) * pixelarea
    ax.plot([2 * x for x in range(len(npolars))], npolars, label=names[i])

ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_ylabel("$N_{polars}$")
ax.set_xlabel("t [ps]")
ax.legend()
fig.savefig(args.out, bbox_inches="tight")
