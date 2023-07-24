# Classes to time evolve initial condition according to the Schrödinger equation
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.linalg import expm, block_diag
import scipy.sparse as sps

class PeriodicSim:
    """
    Solver for periodic x & y boundary conditions using fft.
    Does not incorporate a potential yet. On the bright side, this
    makes the step method accurate for arbitrarily large time steps.

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
    def __init__(self, psi0, startX, endX, startY, endY):
        """
        :param numpy.ndarray psi0: Initial wavefunction in R-basis.
        :param float startx: lowest x value
        :param float starty: lowest y value
        :param float endx: highest x value
        :param float endy: highest y value
        """
        samplesX, samplesY = np.shape(psi0)
        self.psi = psi0
        self._dx = (endX - startX)/samplesX
        self._dy = (endY - startY)/samplesY
        k0x = np.pi/self._dx
        k0y = np.pi/self._dy
        kx = fftshift(np.linspace(-k0x, k0x, samplesX))
        ky = fftshift(np.linspace(-k0y, k0y, samplesY))
        self._kxv, self._kyv = np.meshgrid(kx, ky)
        self.t = 0

    def step(self, dt):
        """Performs one step of time evolution with time step dt."""
        psik = fft2(self.psi)
        psik = psik*np.exp(-0.5j * (self._kxv * self._kxv \
                                    + self._kyv * self._kyv) * dt)
        self.psi = ifft2(psik)
        self.t += dt

class DirSim:
    """
    Non-sparse solver for Dirichlet boundary conditions, i.e. infinite well.
    Uses a baked-in Hamiltonian and time step created at instantiation.
    Uses a 9 point stencil approximation for the Laplace operator
    assuming an evenly spaced square grid.
    """
    def __init__(self, psi0, start, end, dt):
        self._samples = np.shape(psi0)[0]
        self._psi = psi0.flatten()
        self._h = (end - start)/self._samples
        blockOne = np.diag(np.full(self._samples, -3))\
            + np.diag(np.full(self._samples-1, 0.5), 1)\
            + np.diag(np.full(self._samples-1, 0.5), -1)
        
        blockTwo = np.diag(np.full(self._samples, 0.5))\
            + np.diag(np.full(self._samples-1, 0.25), -1)\
            + np.diag(np.full(self._samples-1, 0.25), 1)
        
        offDiagTemplate = np.diag(np.full(self._samples-1, 1), -1)\
            + np.diag(np.full(self._samples-1, 1), 1)
        offDiagH = np.kron(offDiagTemplate, blockTwo)
        diagH = np.kron(np.eye(self._samples), blockOne)
        H = (diagH + offDiagH)/(self._h*self._h)
        self.U = expm(-1.0j*H*dt)
        self.dt = dt
        self.t = 0

    def step(self):
        self._psi = self.U @ self._psi
        self.t += self.dt

    @property
    def psi(self):
        return np.reshape(self._psi, (self._samples, self._samples))

    @psi.setter
    def psi(self, psi0):
        self._psi = psi0.flatten()

class SparseDirSim:
    """
    WIP sparse solver for Dirichlet boundary conditions, i.e. infinite well.
    Uses a baked-in Hamiltonian and time step.
    Uses a 9 point stencil approximation for the Laplace operator
    assuming an evenly spaced square grid.
    """
    def __init__(self, psi0, start, end, dt):
        self._samples = np.shape(psi0)[0]
        self._psi = psi0.flatten()
        self._h = (end - start)/self._samples
        blockOne = sps.diags([0.5, -3, 0.5],
                             [-1, 0, 1],
                             shape=(self._samples, self._samples))
        blockTwo = sps.diags([0.25, 0.5, 0.25],
                             [-1, 0, 1],
                             shape=(self._samples, self._samples))
        offDiagTemplate = sps.diags([1, 0, 1],
                             [-1, 0, 1],
                             shape=(self._samples, self._samples))
        offDiagH = sps.kron(offDiagTemplate, blockTwo)
        diagH = sps.block_diag([blockOne for _ in range(self._samples)])
        H = (diagH + offDiagH)/(self._h*self._h)
        nEigValues = self._samples//10
        w, self._v = sps.linalg.eigsh(H, k=nEigValues)
        self._vinv = self._v.conjugate()
        self._U = np.exp(-1.0j*w*dt)
        self.dt = dt
        self.t = 0

    def step(self):
        self._psi = self._vinv @ (self._U * (self._v.T @ self._psi))
        self.t += self.dt

    @property
    def psi(self):
        return np.reshape(self._psi, (self._samples, self._samples))

    @psi.setter
    def psi(self, psi0):
        self._psi = psi0.flatten()
