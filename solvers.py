# Classes to time evolve initial condition according to the Schrödinger equation
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftshift
from scipy.linalg import expm
import scipy.sparse as sps

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
    def __init__(self, psi0, xv, yv, m, V):
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
        dx = xv[0,0] - xv[0,1]
        dy = yv[0,0] - yv[1,0]
        k0x = np.pi/dx
        k0y = np.pi/dy
        dkx = 2*k0x/samplesX
        dky = 2*k0y/samplesY
        kx = fftshift(np.arange(-k0x, k0x, dkx))
        ky = fftshift(np.arange(-k0y, k0y, dky))
        self._kxv, self._kyv = np.meshgrid(kx, ky)
        self.t = 0

    def step(self, dt):
        """Performs one step of time evolution with time step dt."""
        psik = fft2(np.exp(-1.0j*dt*self.V(self.xv, self.yv, self.psi)/2)*self.psi)
        psik = psik*np.exp(-1.0j * (self._kxv * self._kxv \
                                    + self._kyv * self._kyv) * dt\
                           / (2 * self.m))
        self.psi = np.exp(-1.0j*dt*self.V(self.xv, self.yv, self.psi)/2)*ifft2(psik)
        self.t += dt


class DirSim:
    """
    Non-sparse semi-analytic solver (i.e. expm based).
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
    WIP sparse semi-analytic solver (i.e. expm based).
    Uses a baked-in Hamiltonian and time step.
    Uses a 9 point stencil approximation for the Laplace operator
    assuming an evenly spaced square grid.
    """
    def __init__(self, psi0, start, end, dt):
        self._samples = np.shape(psi0)[0]
        self._psi = psi0.flatten()
        self._h = (end - start)/self._samples
        blockOne = sps.diags([0.5, -3, 0.5],
                             offsets=[-1, 0, 1],
                             shape=(self._samples, self._samples))
        blockTwo = sps.diags([0.25, 0.5, 0.25],
                             offsets=[-1, 0, 1],
                             shape=(self._samples, self._samples))
        offDiagTemplate = sps.diags([1, 0, 1],
                             offsets=[-1, 0, 1],
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

class PeriodicSim1D:
    """
    1D time evolution according to the Schrödinger equation with the
    fft method, giving periodic boundary conditions.
    """
    def __init__(self, psi0, start, end):
        self.samples = len(psi0)
        self.dx = (end - start)/self.samples
        self.k0 = np.pi/self.dx
        self.dk = 2*self.k0/self.samples
        self.k = fftshift(np.arange(-self.k0, self.k0, self.dk))
        self.psi = psi0
    def step(self, dt):
        fy = fft(self.psi) * np.exp(-1.0j*self.k*self.k*dt)
        self.psi = ifft(fy)
    def eventimeevolution(self, t, tsamples):
        dt = t/tsamples
        for _ in range(tsamples):
            self.step(dt)

class DirSim1D:
    """
    1D time evolution according to the Schrödinger equation.
    """
    def __init__(self, psi0, dt, hamiltony, sparse=False):
        self.psi = psi0
        if sparse:
            self.U = sp.linalg.expm(-1.0j*hamiltony*dt)
        else:
            self.U = expm(-1.0j*hamiltony*dt)
        self.sparse = sparse
    def step(self):
        self.psi = self.U @ self.psi
    def nstep(self, n):
        for _ in range(n):
            self.step()
