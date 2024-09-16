import os
import time
from argparse import ArgumentParser
from pathlib import Path
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as tfft
from matplotlib import animation, cm

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

parser = ArgumentParser()
parser.add_argument("file")
parser.add_argument("out")
args = parser.parse_args()

a = np.load(args.file)

psir
