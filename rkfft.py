from scipy.integrate import RK45
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import cm, animation
# import argparse


@jit(nopython=True)
def gauss(x, y):
    return np.exp(-x * x - y * y)


@jit(nopython=True)
def sqr_mod(x):
    return x.real * x.real + x.imag * x.imag


@jit(nopython=True)
def exactsol(x, y, t, a, m):
    return (a / (a + 1.0j * t / m)) \
        * np.exp(-(x * x + y * y) / (2 * (a + 1.0j * t / m)))


startx = -5
endx = 5
samples = 256
dt = 0.01
nframes = 120
times = np.arange(0, nframes*dt, dt)
dx = (endx - startx) / samples
x = np.arange(startx, endx, dx)
xv, yv = np.meshgrid(x, x)
k0x = np.pi / dx
dkx = 2 * k0x / samples
kx = np.fft.fftshift(np.arange(-k0x, k0x, dkx))
kxv, kxy = np.meshgrid(kx, kx)


def f(t, y):
    tmp = np.fft.fft2(np.reshape(y, (samples, samples)))
    tmp = -1.0j*(kxv * kxv + kxy * kxy) * tmp
    return np.reshape(np.fft.ifft2(tmp), (samples*samples))


psi0 = gauss(xv, yv).astype('complex')
sims = [psi0.flatten()]

for i, t in enumerate(times):
    sim = RK45(f, 0, sims[i], dt)
    while sim.status == 'running':
        sim.step()
    sims.append(sim.y)

fig, (dax, eax, sax) = plt.subplots(3)
epsi = exactsol(xv, yv, 0, 0.5, 0.5)
diff = sqr_mod(epsi - np.reshape(sims[0], (samples, samples)))
dim = dax.imshow(diff,
                 cmap=cm.viridis,
                 origin='lower',
                 extent=[startx, endx, startx, endx])
fig.colorbar(dim, ax=dax)
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
    dax.set_title(f"$|\psi - \psi'|^2$ at $t = {times[frame]:.3f}$")
    dim.set_data(diff)
    dim.set_clim(vmin, vmax)
    bleh = sqr_mod(exactpsi)
    evmin = np.min(bleh)
    evmax = np.max(bleh)
    eim.set_data(bleh)
    eim.set_clim(evmin, evmax)
    eax.set_title(f"$|\psi|^2$ at t = {times[frame]:.3f}")
    blah = sqr_mod(spsi)
    svmin = np.min(blah)
    svmax = np.max(blah)
    simim.set_data(blah)
    simim.set_clim(svmin, svmax)
    sax.set_title(f"$|\psi'|^2$ at t = {times[frame]:.3f}")
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

anim.save('animations/rkfftexample.mp4', writer=FFwriter)
