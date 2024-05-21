import os

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(nrows=2, ncols=4, sharex='row', sharey='row')
data = {}
sets = ["pn46", "pn86", "pn111", "pn151"]
for s in sets:
    data[s] = np.load(os.path.join("graphs", s, "psidata")).item()

ims = {}
for i, s in enumerate(sets):
    ims[s+'r'] = ax[0, i].imshow(
            data[s]["rpsidata"],
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=extent,
            )
