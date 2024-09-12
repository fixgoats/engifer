import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from src.solvers import npnormSqr

parser = ArgumentParser()
parser.add_argument("file")
parser.add_argument("out")
args = parser.parse_args()

plots = []
a = np.load(args.file)
psir = a["rpsidata"]
extentr = a["extentr"]
fig, ax = plt.subplots()
im = ax.imshow(
    npnormSqr(psir),
    origin="lower",
    interpolation="none",
    extent=extentr,
)
ax.set_title(r"$|\psi_r|^2$")
ax.set_xlabel(r"x [µm]")
ax.set_ylabel(r"y [µm]")

cb = plt.colorbar(im)
fig.savefig(os.path.join(args.out))
