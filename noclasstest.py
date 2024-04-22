import torch
import torch.fft as tfft

constV = -0.5j * 0.2
alpha = 0.0004
G = 0.002
R = 0.016
pumpStrength = 24
sigma = 2
dt = 0.05
Gamma = 0.1
Gammainv = 1.0 / Gamma
pump = 0
eta = 0


def V(psi, nR):
    return (
        constV
        + alpha * (psi.conj() * psi())
        + (G + 0.5j * R) * (nR + eta * Gammainv * pump)
        + 0.5j * R * nR
    )
