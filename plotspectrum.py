from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from src.solvers import hbar, npnormSqr

parser = ArgumentParser()
parser.add_argument("--file")
parser.add_argument("--out")
args = parser.parse_args()

dt = 0.05
emax = hbar * np.pi / dt
start = 512
end = 512 + 64
spectrum = np.fft.fftshift(npnormSqr(np.load(args.file)))
spectrum /= np.max(spectrum)
spectrum = spectrum[start:end]
xmin = 0
xmax = ((end - start) / 512) * emax
fig, ax = plt.subplots()
ax.plot(np.arange(xmin, xmax, emax / 512), spectrum)
fig.savefig(args.out)
