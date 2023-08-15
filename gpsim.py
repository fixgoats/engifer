from solvers import SsfmGP, normSqr, gauss
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from numba import jit
import argparse

ffmpegPath = subprocess.run(["whereis", "ffmpeg"], capture_output=True, encoding='utf8')
plt.rcParams['animation.ffmpeg_path'] = ffmpegPath.stdout.split()[1]
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--animation')
parser.add_argument('-d', '--dispersion')
args = parser.parse_args()


def exactsol(x, y, t, a, m):
    return (a / (a + 1.0j * t / m)) \
        * np.exp(-(x * x + y * y) / (2 * (a + 1.0j * t / m)))


startX = -20
endX = 20
startY = -20
endY = 20
samples = 256
dx = (endX - startX) / samples
x = np.arange(startX, endX, dx)
xv, yv = np.meshgrid(x, x)
k0x = np.pi / dx
# *(1-(xv/5)**6*(yv/5)**6)

dt = 0.01
nframes = 512
fps = 12
t = dt * np.arange(1, nframes+1)
psi0 = np.random.rand(samples, samples).astype('complex')
#psi0 = np.zeros((samples, samples), dtype='complex')
pump = 200 * (gauss(xv - 5, yv).astype('complex') + gauss(xv + 5, yv).astype('complex'))
nR = np.zeros((samples, samples))
alpha = 0.01
gammalp = 2
Gamma = 2
G = 0.1
R = 4
eta = 1

@jit(nopython=True, nogil=True)
def Vlinear(x, y, wavefunction):
    return (x*x + y*y)


@jit(nopython=True, nogil=True)
def V(x, y, wavefunction, nR):
    return (x / 16)**16 + (y / 16)**16 + \
            alpha * normSqr(wavefunction)\
            + G * (nR + eta / Gamma * pump)\
            - 0.5j * (gammalp - R * nR)


gpsim = SsfmGP(psi0, xv, yv, 0.5, V, nR=nR, gamma=Gamma, R=R, pump=pump, dt=0.01)
snapshots = np.zeros((512, 256), dtype='complex')
fig, ax = plt.subplots()
im = ax.imshow(normSqr(gpsim.psi),
               cmap=cm.viridis,
               origin='lower',
               extent=[startX, endX, startY, endY])
fig.colorbar(im, ax=ax)


def init():
    return [im]


def animate_heatmap(frame):
    gpsim.step()
    vmin = np.min(normSqr(gpsim.psi))
    vmax = np.max(normSqr(gpsim.psi))
    ax.set_title(f"t = {gpsim.t:.3f}")
    im.set_data(normSqr(gpsim.psi))
    im.set_clim(vmin, vmax)
    snapshots[frame, :] = gpsim.psi[127, :]
    return [im]


anim = animation.FuncAnimation(fig,
                               animate_heatmap,
                               init_func=init,
                               frames=nframes,
                               blit=False)
FFwriter = animation.FFMpegWriter(fps=fps,
                                  metadata={'copyright': 'Public Domain'})

anim.save(args.animation, writer=FFwriter)
spectrum = np.fft.fft2(snapshots)
plt.cla()
plt.clf()
fig, ax = plt.subplots()
im = ax.imshow(np.log(np.abs(spectrum)),
               cmap=cm.viridis,
               origin='lower',
               aspect='auto',
               extent=[-k0x, k0x, 0, nframes * dt])
fig.colorbar(im, ax=ax)

plt.savefig(args.dispersion)
