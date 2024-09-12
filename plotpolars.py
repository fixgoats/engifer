from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("file")
parser.add_argument("out")
args = parser.parse_args()

a = np.load(args.file)
npolars = a["nPolars"]
fig, ax = plt.subplots()
ax.plot(npolars)
fig.savefig(args.out)
