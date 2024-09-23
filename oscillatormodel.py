import os
import time
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import torch
import torch.fft as tfft

from src.solvers import hbar


def f(psi):
    return (
        p
        + (-1j * alpha - R) * (psi**2 + 2 * psi**2) * psi
        + sum(J * psim + Jb * psim0 * e)
    )
