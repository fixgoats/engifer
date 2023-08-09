from solvers import SsfmGP, normSqr, gauss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from numba import jit
import argparse

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()


def exactsol(x, y, t, a, m):
    return (a / (a + 1.0j * t / m)) \
        * np.exp(-(x * x + y * y) / (2 * (a + 1.0j * t / m)))


startX = -10
endX = 10
startY = -10
endY = 10
samples = 256
dx = (endX - startX) / samples
x = np.arange(startX, endX, dx)
xv, yv = np.meshgrid(x, x)
# *(1-(xv/5)**6*(yv/5)**6)

dt = 0.01
nframes = 200
fps = 12
t = dt * np.arange(1, nframes+1)
psi0 = np.random.rand(samples, samples).astype('complex')
pump = 20 * gauss(xv, yv).astype('complex')
nR = np.zeros((samples, samples))
alpha = 0
gammalp = 1
Gamma = 1
R = 2


@jit(nopython=True, nogil=True)
def Vlinear(x, y, wavefunction):
    return (x*x + y*y)


@jit(nopython=True, nogil=True)
def V(x, y, wavefunction, nR):
    return alpha*normSqr(wavefunction)\
            - 0.5j * (gammalp - R * nR)


gpsim = SsfmGP(psi0, xv, yv, 0.5, V, nR=nR, gamma=Gamma, R=R, pump=pump, dt=0.01)
fig, ax = plt.subplots()
im = ax.imshow(normSqr(gpsim.nR),
               cmap=cm.viridis,
               origin='lower',
               extent=[startX, endX, startY, endY])
fig.colorbar(im, ax=ax)


def init():
    return [im]


def animate_heatmap(frame):
    gpsim.step()
    vmin = np.min(gpsim.nR.real)
    vmax = np.max(gpsim.nR.real)
    ax.set_title(f"t = {gpsim.t:.3f}")
    im.set_data(gpsim.nR.real)
    im.set_clim(vmin, vmax)
    return [im]


anim = animation.FuncAnimation(fig,
                               animate_heatmap,
                               init_func=init,
                               frames=nframes,
                               blit=False)
FFwriter = animation.FFMpegWriter(fps=fps,
                                  metadata={'copyright': 'Public Domain'})

anim.save(args.filename, writer=FFwriter)
