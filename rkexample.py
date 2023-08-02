from scipy.integrate import RK45
# from scipy.linalg import expm
# from scipy.signal import convolve2d
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import cm, animation
import argparse

parser = argparse.ArgumentParser(
        prog='rksim',
        description='runs a RK45 simulation fo the Schrödinger equation')

parser.add_argument('-s', '--save', help='Saves arrays to binary file')
parser.add_argument('-c', '--use-cache', help='Uses specified arrays to go\
                                                straight to animating')
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


@jit(nopython=True)
def gauss(x, y):
    return np.exp(-x * x - y * y)


@jit(nopython=True)
def gauss1d(x):
    return np.exp(-x*x)


@jit(nopython=True)
def sqr_mod(x):
    return x.real * x.real + x.imag * x.imag


@jit(nopython=True)
def exactsol(x, y, t, a, m):
    return (a / (a + 1.0j * t / m)) \
        * np.exp(-(x * x + y * y) / (2 * (a + 1.0j * t / m)))


@jit(nopython=True)
def exactsol1d(x, t, a, m):
    return np.sqrt((a / (a + 1.0j * t / m))) \
        * np.exp(-(x * x) / (2 * (a + 1.0j * t / m)))


startx = -5
endx = 5

samples = 256
dx = (endx - startx) / samples
dt = 0.01
nframes = 120
times = np.arange(0, nframes*dt, dt)
x = np.arange(startx, endx, dx)
# x = np.linspace(startx, endx, samples)
xv, yv = np.meshgrid(x, x)
# for 2d case, "exact"
# blockOne = np.diag(np.full(samples, 3))\
#    + np.diag(np.full(samples-1, -0.5), 1)\
#    + np.diag(np.full(samples-1, -0.5), -1)
#
# blockTwo = np.diag(np.full(samples, -0.5))\
#    + np.diag(np.full(samples-1, -0.25), -1)\
#    + np.diag(np.full(samples-1, -0.25), 1)
#
# offDiagTemplate = np.diag(np.full(samples-1, 1), -1)\
#    + np.diag(np.full(samples-1, 1), 1)
# offDiagH = np.kron(offDiagTemplate, blockTwo)
# diagH = np.kron(np.eye(samples), blockOne)
blockOne = sps.diags([-0.5, 3, -0.5],
                     offsets=[-1, 0, 1],
                     shape=(samples, samples))
blockTwo = sps.diags([-0.25, -0.5, -0.25],
                     offsets=[-1, 0, 1],
                     shape=(samples, samples))
# blockOne = sps.diags([-1, 4, -1],
#                      offsets=[-1, 0, 1],
#                      shape=(samples,samples))
# blockTwo = sps.diags([-1],
#                     shape=(samples,samples))
offDiagTemplate = sps.diags([1, 1], offsets=[-1, 1], shape=(samples, samples))
offDiagH = sps.kron(offDiagTemplate, blockTwo)
diagH = sps.block_diag([blockOne for _ in range(samples)])
modH = -1.0j * (diagH + offDiagH)/(dx*dx)
# H = (diagH + offDiagH)/(dx*dx)
psi0 = gauss(xv, yv).astype('complex')
sims = [psi0.flatten()]


def f(t, y):
    return modH @ y


for i, t in enumerate(times):
    sim = RK45(f, 0, sims[i], dt)
    while sim.status == 'running':
        sim.step()
    sims.append(sim.y)
# for 1d case
# H = (np.diag(np.full(samples, 2))\
#         + np.diag(-np.ones(samples-1), 1)\
#         + np.diag(-np.ones(samples-1), -1))/(dx*dx)
# psi0 = gauss1d(x)
# epsi = exactsol1d(x, 0.5, 0.5, 0.5)
# U = expm(-1.0j*H*dt)
# psi = U @ psi0.flatten()

fig, (dax, eax, sax) = plt.subplots(3)
epsi = exactsol(xv, yv, 0, 0.5, 0.5)
diff = sqr_mod(epsi - np.reshape(sims[0], (samples, samples)))
dim = dax.imshow(diff,
                 cmap=cm.viridis,
                 origin='lower',
                 extent=[startx, endx, startx, endx])
fig.colorbar(dim, ax=eax)
eim = eax.imshow(sqr_mod(epsi),
                 cmap=cm.viridis,
                 origin='lower',
                 extent=[startx, endx, startx, endx])
fig.colorbar(eim, ax=eax)
simim = sax.imshow(sqr_mod(psi0),
                   cmap=cm.viridis,
                   origin='lower',
                   extent=[startx, endx, startx, endx])
fig.colorbar(simim, ax=sax)
# fig, ax = plt.subplots(2, 2)
# im1 = ax[0,0].imshow(np.reshape(sim.y.real, (samples, samples)),
#                cmap=cm.viridis,
#                origin='lower',
#                extent=[startx, endx, startx, endx])
# fig.colorbar(im1, ax=ax[0,0])
# im2 = ax[0,1].imshow(np.reshape(sim.y.imag, (samples, samples)),
#                cmap=cm.viridis,
#                origin='lower',
#                extent=[startx, endx, startx, endx])
# fig.colorbar(im2, ax=ax[0,1])
# im3 = ax[1,0].imshow(epsi.real,
#                cmap=cm.viridis,
#                origin='lower',
#                extent=[startx, endx, startx, endx])
# fig.colorbar(im3, ax=ax[1,0])
# im4 = ax[1,1].imshow(epsi.imag,
#                cmap=cm.viridis,
#                origin='lower',
#                extent=[startx, endx, startx, endx])
# fig.colorbar(im4, ax=ax[1,1])


def init():
    return [dim, eim, simim]


def animate_heatmap(frame):
    exactpsi = exactsol(xv, yv, times[frame], 0.5, 0.5)
    spsi = np.reshape(sims[frame], (samples, samples))
    diff = sqr_mod(exactpsi - spsi)
    vmin = np.min(diff)
    vmax = np.max(diff)
    # ax.set_title(f"sim t = {t[frame]:.3f}")
    dim.set_data(diff)
    dim.set_clim(vmin, vmax)
    bleh = sqr_mod(exactpsi)
    evmin = np.min(bleh)
    evmax = np.max(bleh)
    eim.set_data(bleh)
    eim.set_clim(evmin, evmax)
    bleh = sqr_mod(spsi)
    svmin = np.min(bleh)
    svmax = np.max(bleh)
    simim.set_data(bleh)
    simim.set_clim(svmin, svmax)
    return [dim, eim, simim]


# ax[0,0].plot(x, psi.real)
# ax[0,1].plot(x, psi.imag)
# ax[1,0].plot(x, epsi.real)
# ax[1,1].plot(x, epsi.imag)
# plt.show()

fps = 12
anim = animation.FuncAnimation(fig,
                               animate_heatmap,
                               init_func=init,
                               frames=nframes,
                               blit=True)
FFwriter = animation.FFMpegWriter(fps=fps,
                                  metadata={'copyright': 'Public Domain'})

anim.save('testingdiff.mp4', writer=FFwriter)
