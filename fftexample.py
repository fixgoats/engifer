from solvers import PeriodicSim, PeriodicSim1D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


def gauss(x, y):
    return np.exp(-x * x - y * y)


def sqr_mod(x):
    return x.real * x.real + x.imag * x.imag


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
t = dt * np.arange(1,nframes+1)
psi0 = gauss(xv, yv)
fftsim = PeriodicSim(psi0, startX, endX, startY, endY, 0.5)
fig, ax = plt.subplots()
exactpsi = exactsol(xv, yv, 0, 0.5, 0.5)
diff = exactpsi - fftsim.psi
im = ax.imshow(sqr_mod(diff),
               cmap=cm.viridis,
               origin='lower',
               extent=[startX, endX, startY, endY])
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# surf = ax.plot_surface(xv, yv, sqr_mod(diff), cmap=cm.viridis, antialiased=False)
fig.colorbar(im, ax=ax)
#plt.show()

def init():
    return [im]

def animate_heatmap(frame):
    exactpsi = exactsol(xv, yv, t[frame], 0.5, 0.5)
    fftsim.step(dt)
    diff = sqr_mod(exactpsi - fftsim.psi)
    vmin = np.min(diff)
    vmax = np.max(diff)
    ax.set_title(f"exact t = {t[frame]:.3f}, sim t = {fftsim.t:.3f}")
    im.set_data(diff)
    im.set_clim(vmin, vmax)
    return [im]

# print("animate function has been processed")
# def animate_surface_diff(frame):
#     fftsim.step(dt)
#     exactpsi = exactsol(xv, yv, t[frame], 0.5, 0.5)
#     diff = exactpsi - fftsim.psi
#     ax.clear()
#     surf = ax.plot_surface(xv,
#                            yv,
#                            sqr_mod(diff),
#                            cmap=cm.viridis,
#                            antialiased=False)
#     surf._facecolors2d = surf._facecolor3d
#     surf._edgecolors2d = surf._facecolor3d
#     return [surf]

# def animate_surface_fft(frame):
#     fftsim.step(dt)
#     ax.clear()
#     surf = ax.plot_surface(xv,
#                            yv,
#                            sqr_mod(fftsim.psi),
#                            cmap=cm.viridis,
#                            antialiased=False)
#     surf._facecolors2d = surf._facecolor3d
#     surf._edgecolors2d = surf._edgecolor3d
#     return [surf]


# def animate_surface_exact(frame):
#     exactpsi = exactsol(xv, yv, t[frame], 0.5, 0.5)
#     ax.clear()
#     surf = ax.plot_surface(xv,
#                            yv,
#                            sqr_mod(exactpsi),
#                            cmap=cm.viridis,
#                            antialiased=False)
#     surf._facecolors2d = surf._facecolor3d
#     surf._edgecolors2d = surf._edgecolor3d
#     return [surf]

anim = animation.FuncAnimation(fig,
                               animate_heatmap,
                               init_func=init,
                               frames=nframes,
                               blit=False)
FFwriter = animation.FFMpegWriter(fps=fps,
                                  metadata={'copyright': 'Public Domain'})

anim.save('testingdiff.mp4', writer=FFwriter)
