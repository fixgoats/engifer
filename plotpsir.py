from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

a = np.load(args.file, allow_pickle=True).item()
psir = a["rpsidata"]
extentr = a["extentr"]
fig, ax = plt.subplots()
im = ax.imshow(psir, origin="lower", interpolation="none", extent=extentr)
cb = plt.colorbar(im)
plt.show()
