import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("--file", required=False)
parser.add_argument("--set", required=False)
args = parser.parse_args()

if args.file is not None:
    a = np.load(args.file, allow_pickle=True).item()
    spectra = a["spectra"]
    extent = a["extent"]
    starty = int(np.shape(spectra)[0] * 0.5)
    endy = int(np.shape(spectra)[0] * 0.52)
    extent[2] = 0
    extent[3] /= 25

    fig, ax = plt.subplots()
    im = ax.imshow(
        spectra[starty:endy, :],
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=extent,
    )
    cb = plt.colorbar(im)
    plt.show()

if args.set is not None:
    setups = [
        "thick",
        "thin",
        "thickthick",
        "thinthin",
        "thickthin",
        "thinthickthin",
        "penrose0",
        "penrose1",
        "penrose2",
    ]
    spectra = {}
    for name in setups:
        spectra[name] = np.load(
            os.path.join("graphs", name, args.set, "spectra.npy"), allow_pickle=True
        ).item()["spectra"]
        starty = int(np.shape(spectra[name])[0] * 0.5)
        endy = int(np.shape(spectra[name])[0] * 0.55)
        spectra[name] = spectra[name][starty:endy, :]

    extent = np.load(
        os.path.join("graphs/thick", args.set, "spectra.npy"), allow_pickle=True
    ).item()["extent"]
    extent[2] = 0
    extent[3] /= 10
    fig, ax = plt.subplots(nrows=len(setups), sharex=True)
    fig.set_size_inches(w=6.4, h=16)
    axes = {}
    for i, name in enumerate(setups):
        axes[name] = ax[i]

    ims = {}
    for key in axes:
        ims[key] = axes[key].imshow(
            spectra[key],
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=extent,
        )
        axes[key].set_title(key)
    fig.subplots_adjust(left=0.1, bottom=0.02, top=0.98, right=0.85)
    cbar_ax = fig.add_axes([0.90, 0.02, 0.03, 0.96])
    # axes["penrose2"].set_xlim(xmin=extent[0], xmax=extent[1])
    fig.colorbar(ims["thick"], cax=cbar_ax)
    basedir = os.path.join("graphs", args.set)
    Path(basedir).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(basedir, "uhh.pdf"))
