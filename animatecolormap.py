import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

parser = ArgumentParser()
parser.add_argument("-i", "--infile")
parser.add_argument("-o", "--outfile")
args = parser.parse_args()

fig, ax = plt.subplots()
animationarray = np.load(os.path.join(args.infile))

im = ax.imshow(
    animationarray[:, :, 0] / np.max(animationarray[:, :, 0]),
    origin="lower",
    interpolation="none",
)

cb = fig.colorbar(im)


def update(frame):
    im.set_array(
        animationarray[:, :, frame] / max(np.max(animationarray[:, :, frame]), 1e-4)
    )
    return [im]


anim = FuncAnimation(
    fig,
    update,
    frames=np.shape(animationarray)[2],
    blit=True,
)
FFwriter = FFMpegWriter(fps=24, metadata={"copyright": "Public Domain"})
anim.save(args.outfile, writer=FFwriter)
