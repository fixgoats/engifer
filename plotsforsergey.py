from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from src.solvers import npnormSqr

parser = ArgumentParser()
parser.add_argument("--file")
parser.add_argument("--out")
args = parser.parse_args()

a = np.load(args.file, allow_pickle=True).item()
psir = a["psir"]
fig, ax = plt.subplots()
im = ax.imshow(npnormSqr(psir), origin="lower", interpolation="none")
cb = plt.colorbar(im)
if args.out is None:
    plt.show()

else:
    fig.savefig(args.out)
