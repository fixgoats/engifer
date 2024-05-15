import os

import matplotlib.pyplot as plt
import numpy as np

a = np.load(os.path.join("graphs", "thick", "2024-05-03", "16.26", "spectra.npy"), allow_pickle=True).item()
fig, ax = plt.subplots()
spectra = a["spectra"]
starty = int(np.shape(spectra)[0] * 0.5)
endy = int(np.shape(spectra)[0] * 0.55)
spectra = spectra[starty:endy, :]
extent = [10, 24, 0, a["extent"][3] / 10]
print(np.shape(spectra))
im = ax.imshow(spectra, origin="lower", interpolation="none", extent=extent, aspect="auto")
plt.show()
plt.close()

fig, ax = plt.subplots()
a = np.load(os.path.join("graphs", "thick", "2024-05-03", "16.26", "psidata24.00.npy"), allow_pickle=True)
im = ax.imshow(a.item()["rpsidata"], origin="lower", interpolation="none", extent=extent, aspect="auto")
plt.colorbar(im)
plt.show()
plt.close()
