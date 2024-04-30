# Classes to time evolve initial condition according to the Schrödinger equation
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import torch
import torch.fft as tfft
from numba import complex128, float64, vectorize
from scipy.fft import fft, fft2, fftshift, ifft, ifft2
from scipy.linalg import expm
from scipy.signal import convolve2d

hbar = 6.582119569e-1  # meV * ps


def figBoilerplate():
    plt.cla()
    fig, ax = plt.subplots()
    fig.dpi = 300
    fig.figsize = (6.4, 4.8)
    return fig, ax


def imshowBoilerplate(
    data, filename, xlabel="", ylabel="", extent=[], title="", aspect="auto"
):
    fig, ax = figBoilerplate()
    im = ax.imshow(
        data, aspect=aspect, origin="lower", interpolation="none", extent=extent
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(filename)
    plt.close()
    print(f"Made plot {filename}")


# Using numba for this is actually faster than the default, but vectorizing
# with numpy is somehow slower than default. This could be made even faster by using njit
# with known array dimensions, but then you'd need distinct functions for vectors
# and matrices.
@vectorize([float64(complex128)])
def npnormSqr(x):
    return x.real * x.real + x.imag * x.imag


# For use with torch. This case doesn't benefit from jit and such and
# the conj version is just as fast as the expanded version
def tnormSqr(x):
    return x.conj() * x


# Create a somewhat more realistic initial condition than white noise
def smoothnoise(xv, yv):
    random = np.random.uniform(-1, 1, np.shape(xv)) + 1j * np.random.uniform(
        -1, 1, np.shape(xv)
    )
    krange = np.linspace(-2, 2, num=21)
    kbasex, kbasey = np.meshgrid(krange, krange)
    kernel = gauss(kbasex, kbasey)
    kernel /= np.sum(kernel)
    output = convolve2d(random, kernel, mode="same")
    output = convolve2d(output, kernel, mode="same")
    return output / np.sqrt(np.sum(npnormSqr(output)))


def gauss(x, y, sigmax=1, sigmay=1):
    return np.exp(-((x / sigmax) ** 2) - (y / sigmay) ** 2)


def tgauss(x, y, sigmax=1, sigmay=1):
    return torch.exp(-((x / sigmax) ** 2) - (y / sigmay) ** 2)


class PeriodicSim:
    """
    Solver for periodic x & y boundary conditions using fft.
    Does not incorporate a potential yet. On the bright side, this
    makes the step method accurate for arbitrarily large time steps.
    Or it would be if the solution didn't slowly move diagonally when
    it should be in one place.

    Attributes
    ----------
    :attribute numpy.ndarray psi: 2d numpy array
        Wavefunction in R-basis. Assumes evenly spaced samples where
        x varies within rows and y varies within columns (or maybe
        it's the other way around). For best performance this should
        be a 2^n by 2^m matrix for some n, m. Running the simulation
        modifies this attribute.
    :attribute float t: Time elapsed since start of emulation.
    Automatically managed.
    """

    def __init__(self, psi0, xv, yv, m, V, dt):
        """
        :param numpy.ndarray psi0: Initial wavefunction in R-basis.
        :param float startX: lowest x value
        :param float startY: lowest y value
        :param float endX: highest x value
        :param float endY: highest y value
        """
        samplesX, samplesY = np.shape(psi0)
        self.psi = psi0
        self.m = m
        self.V = V
        self.xv = xv
        self.yv = yv
        dx = xv[0, 0] - xv[0, 1]
        dy = yv[0, 0] - yv[1, 0]
        k0x = np.pi / dx
        k0y = np.pi / dy
        dkx = 2 * k0x / samplesX
        dky = 2 * k0y / samplesY
        kx = fftshift(np.arange(-k0x, k0x, dkx))
        ky = fftshift(np.arange(-k0y, k0y, dky))
        self._kxv, self._kyv = np.meshgrid(kx, ky)
        self.t = 0
        self.dt = dt
        self.kTimeEvo = np.exp(
            -1.0j * (self._kxv * self._kxv + self._kyv * self._kyv) * dt / (2 * self.m)
        )

    def step(self):
        """Performs one step of time evolution with time step dt."""
        rTimeEvo = np.exp(-1.0j * self.dt * self.V(self.xv, self.yv, self.psi) / 2)
        psik = fft2(rTimeEvo * self.psi)
        psik = psik * self.kTimeEvo
        self.psi = rTimeEvo * ifft2(psik)
        self.t += self.dt


class SsfmGPGPU:
    __slots__ = (
        "dev",
        "psi",
        "n",
        "psik",
        "kxv",
        "kyv",
        "gridX",
        "gridY",
        "m",
        "nR",
        "dt",
        "t",
        "R",
        "pump",
        "Gamma",
        "Gammainv",
        "m",
        "kTimeEvo",
        "alpha",
        "G",
        "eta",
        "constV",
        "constpart",
    )

    def __init__(
        self,
        dev,
        psi0,
        gridX,
        gridY,
        m,
        nR0,
        alpha,
        Gamma,
        gammalp,
        R,
        pump,
        G,
        eta,
        constV,
        dt,
    ):
        self.psi = psi0.type(dtype=torch.cfloat).to(device=dev)
        self.n = gridX.shape[0]
        self.psik = tfft.fft2(self.psi)
        self.gridX = gridX.type(dtype=torch.cfloat).to(device=dev)
        self.gridY = gridY.type(dtype=torch.cfloat).to(device=dev)
        dx = gridX[0, 1] - gridX[0, 0]
        dy = gridY[1, 0] - gridY[0, 0]
        kxmax = np.pi / dx
        kymax = np.pi / dy
        dkx = 2 * kxmax / psi0.size(dim=0)
        dky = 2 * kymax / psi0.size(dim=1)
        kx = torch.arange(-kxmax, kxmax, dkx)
        ky = torch.arange(-kymax, kymax, dky)
        kx = tfft.fftshift(kx)
        ky = tfft.fftshift(ky)
        kxv, kyv = torch.meshgrid(kx, ky, indexing="ij")
        self.kxv = kxv.to(device=dev)
        self.kyv = kyv.to(device=dev)
        self.t = 0
        self.dt = dt
        self.nR = nR0.type(dtype=torch.cfloat).to(device=dev)
        self.R = R
        self.pump = pump.type(dtype=torch.cfloat).to(device=dev)
        self.Gamma = Gamma
        self.Gammainv = 1.0 / Gamma
        self.m = m
        squareK = kxv * kxv + kyv * kyv
        self.kTimeEvo = torch.exp(-0.5j * hbar * squareK * dt / m).to(device=dev)
        self.alpha = alpha
        self.eta = eta
        self.G = G
        self.constV = constV.to(device=dev)
        self.constpart = self.constV + G * eta * self.Gammainv * self.pump

    def updatePump(self, pump, dev):
        self.pump = pump
        self.constpart = (self.constV + self.G * self.eta * self.Gammainv * pump).to(
            device=dev
        )

    def V(self):
        return (
            self.constpart
            + self.alpha * self.psiNormSqr()
            + (self.G + 0.5j * self.R) * self.nR
        )

    def halfRStepPsi(self):
        halfRTimeEvo = torch.exp(-0.5j * self.dt * self.V())
        self.psi = self.psi * halfRTimeEvo

    def halfRStepNR(self):
        halfRTimeEvo = torch.exp(
            -0.5 * self.dt * (self.Gamma + self.R * self.psiNormSqr())
        )
        self.nR = halfRTimeEvo * self.nR + self.pump * self.dt * 0.5

    def psiNormSqr(self):
        return self.psi.conj() * self.psi

    def psikNormSqr(self):
        return tfft.fftshift(self.psik.conj()) * tfft.fftshift(self.psik)

    def step(self):
        self.halfRStepPsi()
        tfft.fft2(self.psi, out=self.psik)
        self.psik = self.psik * self.kTimeEvo
        tfft.ifft2(self.psik, out=self.psi)
        self.halfRStepNR()
        self.halfRStepPsi()
        self.halfRStepNR()
        self.t += self.dt

    def angmom(self):
        psiykx = tfft.fftshift(tfft.fft(tfft.fftshift(self.psi, dim=0), dim=0), dim=0)
        psixky = tfft.fftshift(tfft.fft(tfft.fftshift(self.psi, dim=1), dim=1), dim=1)
        return (
            torch.sum(
                psixky.conj() * self.gridX * self.kyv * psixky
                - psiykx.conj() * self.gridY * self.kxv * psiykx
            )
            / self.n
        )
