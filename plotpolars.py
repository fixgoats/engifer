from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("--file")
parser.add_argument("--out")
args = parser.parse_args()

npolars = np.load(args.file)
fig, ax = plt.subplots()
ax.plot(npolars)
fig.savefig(args.out)
