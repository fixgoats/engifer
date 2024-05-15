from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

npolars = np.load(args.file, allow_pickle=True)
fig, ax = plt.subplots()
ax.plot(npolars)
plt.show()
