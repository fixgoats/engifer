# Classes to time evolve initial condition according to the Schrödinger equation
import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftshift
from scipy.linalg import expm
import scipy.sparse as sps
from numba import jit
import torch
import torch.fft as tfft


@jit(nopython=True, nogil=True)
def normSqr(x):
    return x.real * x.real + x.imag * x.imag


@jit(nopython=True, nogil=True)
def gauss(x, y, a=1, scale=1):
    return scale * np.exp(-(x * x + y * y)/a)


def findLocalMaxima(vector, fuzz):
    idxValPairs = []
    last = vector[0]
    ascending = True
    for i, val in enumerate(vector):
        if ascending and abs(val - last) > fuzz and val < last:
            idxValPairs.append((i-1, last.item()))
            ascending = False
        elif not ascending and abs(val - last) > fuzz and val >= last:
            ascending = True

        last = val
    idxValPairs = sorted(idxValPairs, key=lambda x: x[1])
    return idxValPairs


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
        k0x = np.pi/dx
        k0y = np.pi/dy
        dkx = 2*k0x/samplesX
        dky = 2*k0y/samplesY
        kx = fftshift(np.arange(-k0x, k0x, dkx))
        ky = fftshift(np.arange(-k0y, k0y, dky))
        self._kxv, self._kyv = np.meshgrid(kx, ky)
        self.t = 0
        self.dt = dt
        self.kTimeEvo = np.exp(-1.0j *
                               (self._kxv * self._kxv
                                + self._kyv * self._kyv) * dt
                               / (2 * self.m))

    def step(self):
        """Performs one step of time evolution with time step dt."""
        rTimeEvo = np.exp(-1.0j*self.dt*self.V(self.xv, self.yv, self.psi)/2)
        psik = fft2(rTimeEvo * self.psi)
        psik = psik*self.kTimeEvo
        self.psi = rTimeEvo*ifft2(psik)
        self.t += self.dt


class SsfmGPNp:
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
        it's the other way around?). For best performance this should
        be a 2^n by 2^m matrix for some n, m. Running the simulation
        modifies this attribute.
    :attribute float t: Time elapsed since start of emulation.
    Automatically managed.
    """

    def __init__(self, psi0, xv, yv, m, V, nR, gamma, R, pump, dt):
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
        k0x = np.pi/dx
        k0y = np.pi/dy
        dkx = 2*k0x/samplesX
        dky = 2*k0y/samplesY
        kx = fftshift(np.arange(-k0x, k0x, dkx))
        ky = fftshift(np.arange(-k0y, k0y, dky))
        self._kxv, self._kyv = np.meshgrid(kx, ky)
        self.t = 0
        self.dt = dt
        self.nR = nR
        self.R = R
        self.pump = pump
        self.gamma = gamma
        self.kTimeEvo = np.exp(-1.0j *
                               (self._kxv * self._kxv
                                + self._kyv * self._kyv) * dt
                               / (2 * self.m))

    def step(self):
        """Performs one step of time evolution with time step dt."""
        rTimeEvo = np.exp(-0.5j*self.dt*self.V(self.xv, self.yv, self.psi, self.nR))
        psik = fft2(rTimeEvo * self.psi)
        psik = psik*self.kTimeEvo
        self.psi = rTimeEvo*ifft2(psik)
        self.nR = np.exp(-(self.gamma + self.R*normSqr(self.psi))*self.dt)\
            * self.nR + self.pump * self.dt
        self.t += self.dt


class SsfmGPCUDA:
    __slots__ = ('psi',
                 'psik',
                 'kxv',
                 'kyv',
                 'm',
                 'nR',
                 'dt',
                 't',
                 'R',
                 'pump',
                 'Gamma',
                 'm',
                 'kTimeEvo',
                 'alpha',
                 'G',
                 'eta',
                 'constV')

    def __init__(self,
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
                 dt):
        cuda = torch.device('cuda')
        self.psi = psi0.type(dtype=torch.cfloat).to(device=cuda)
        self.psik = tfft.fft2(self.psi)
        dx = gridX[0, 1] - gridX[0, 0]
        dy = gridY[1, 0] - gridY[0, 0]
        kxmax = np.pi / dx
        kymax = np.pi / dy
        dkx = 2*kxmax / psi0.size(dim=0)
        dky = 2*kymax / psi0.size(dim=1)
        kx = torch.arange(-kxmax, kxmax, dkx)
        ky = torch.arange(-kymax, kymax, dky)
        kx = tfft.fftshift(kx)
        ky = tfft.fftshift(ky)
        kxv, kyv = torch.meshgrid(kx, ky, indexing='ij')
        self.t = 0
        self.dt = dt
        self.nR = nR0.type(dtype=torch.cfloat).to(device=cuda)
        self.R = R
        self.pump = pump.type(dtype=torch.cfloat).to(device=cuda)
        self.Gamma = Gamma
        self.m = m
        squareK = kxv * kxv + kyv * kyv
        self.kTimeEvo = torch.exp(-1.0j * squareK * dt / (2 * m)).to(device=cuda)
        self.alpha = alpha
        self.eta = eta
        self.G = G
        self.constV = constV.to(device=cuda)

    def V(self):
        return self.constV + self.alpha * self.psiNormSqr() \
                + self.G * (self.nR + self.eta * self.pump / self.Gamma)\
                + 0.5j * self.R * self.nR

    def halfRStepPsi(self):
        halfRTimeEvo = torch.exp(-0.5j * self.dt * self.V())
        self.psi = self.psi * halfRTimeEvo

    def halfRStepNR(self):
        halfRTimeEvo = torch.exp(-0.5 * self.dt * (self.Gamma + self.R * self.psiNormSqr()))
        self.nR = halfRTimeEvo * self.nR + self.pump * self.dt / 2

    def psiNormSqr(self):
        return self.psi.conj() * self.psi

    def psikNormSqr(self):
        return tfft.fftshift(self.psik.conj()) * tfft.fftshift(self.psik)

    def step(self):
        self.halfRStepPsi()
        tfft.fft2(self.psi, out=self.psik, norm='ortho')
        self.psik = self.psik * self.kTimeEvo
        tfft.ifft2(self.psik, out=self.psi, norm='ortho')
        self.halfRStepNR()
        self.halfRStepPsi()
        self.halfRStepNR()
        self.t += self.dt

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
