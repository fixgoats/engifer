from scipy.integrate import RK45
# from scipy.linalg import expm
# from scipy.signal import convolve2d
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import cm, animation

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

samples = 512 
dx = (endx - startx) / samples
t = 0.1
x = np.arange(startx, endx, dx)
#x = np.linspace(startx, endx, samples)
xv, yv = np.meshgrid(x, x)
# for 2d case, "exact"
#blockOne = np.diag(np.full(samples, 3))\
#    + np.diag(np.full(samples-1, -0.5), 1)\
#    + np.diag(np.full(samples-1, -0.5), -1)
#
#blockTwo = np.diag(np.full(samples, -0.5))\
#    + np.diag(np.full(samples-1, -0.25), -1)\
#    + np.diag(np.full(samples-1, -0.25), 1)
#
#offDiagTemplate = np.diag(np.full(samples-1, 1), -1)\
#    + np.diag(np.full(samples-1, 1), 1)
#offDiagH = np.kron(offDiagTemplate, blockTwo)
#diagH = np.kron(np.eye(samples), blockOne)
blockOne = sps.diags([-0.5, 3, -0.5],
                     offsets=[-1, 0, 1],
                     shape=(samples, samples))
blockTwo = sps.diags([-0.25, -0.5, -0.25],
                     offsets=[-1, 0, 1],
                     shape=(samples, samples))
#blockOne = sps.diags([-1, 4, -1],
#                     offsets=[-1, 0, 1],
#                     shape=(samples,samples))
#blockTwo = sps.diags([-1],
#                     shape=(samples,samples))
offDiagTemplate = sps.diags([1, 1],
                     offsets=[-1, 1],
                     shape=(samples, samples))
offDiagH = sps.kron(offDiagTemplate, blockTwo)
diagH = sps.block_diag([blockOne for _ in range(samples)])
H = (diagH + offDiagH)/(dx*dx)
#H = (diagH + offDiagH)/(dx*dx)
psi0 = gauss(xv, yv).astype('complex')
sims = [(psi0, 0)]
epsi = exactsol(xv, yv, t, 0.5, 0.5)
# 2d case, RK
def f(t, y):
    return -1.0j * H @ y

sim = RK45(f, 0, psi0.flatten(), t)
while sim.status == 'running':
    sim.step()
# for 1d case
# H = (np.diag(np.full(samples, 2))\
#         + np.diag(-np.ones(samples-1), 1)\
#         + np.diag(-np.ones(samples-1), -1))/(dx*dx)
#psi0 = gauss1d(x)
#epsi = exactsol1d(x, 0.5, 0.5, 0.5)
#U = expm(-1.0j*H*dt)
#psi = U @ psi0.flatten()
fig, ax = plt.subplots()
diff = sqr_mod(epsi - np.reshape(sim.y, (samples, samples)))
abs = np.maximum(sqr_mod(np.reshape(sim.y, (samples, samples))), sqr_mod(epsi))
im = ax.imshow(diff/abs,
               cmap=cm.viridis,
               origin='lower',
               extent=[startx, endx, startx, endx])
fig.colorbar(im, ax=ax)
#fig, ax = plt.subplots(2, 2)
#im1 = ax[0,0].imshow(np.reshape(sim.y.real, (samples, samples)),
#               cmap=cm.viridis,
#               origin='lower',
#               extent=[startx, endx, startx, endx])
#fig.colorbar(im1, ax=ax[0,0])
#im2 = ax[0,1].imshow(np.reshape(sim.y.imag, (samples, samples)),
#               cmap=cm.viridis,
#               origin='lower',
#               extent=[startx, endx, startx, endx])
#fig.colorbar(im2, ax=ax[0,1])
#im3 = ax[1,0].imshow(epsi.real,
#               cmap=cm.viridis,
#               origin='lower',
#               extent=[startx, endx, startx, endx])
#fig.colorbar(im3, ax=ax[1,0])
#im4 = ax[1,1].imshow(epsi.imag,
#               cmap=cm.viridis,
#               origin='lower',
#               extent=[startx, endx, startx, endx])
#fig.colorbar(im4, ax=ax[1,1])
#def init():
#    return [im]
#
#def animate_heatmap(frame):
#    print(f"{frame[1]}")
#    exactpsi = exactsol(xv, yv, frame[1], 0.5, 0.5)
#    diff = sqr_mod(exactpsi - frame[0])
#    vmin = np.min(diff)
#    vmax = np.max(diff)
#    ax.set_title(f"sim t = {frame[1]:.3f}")
#    im.set_data(diff)
#    im.set_clim(vmin, vmax)
#    return [im]


#ax[0,0].plot(x, psi.real)
#ax[0,1].plot(x, psi.imag)
#ax[1,0].plot(x, epsi.real)
#ax[1,1].plot(x, epsi.imag)
plt.show()

#nframes = 120
#fps = 12
#anim = animation.FuncAnimation(fig,
#                               animate_heatmap,
#                               init_func=init,
#                               frames=sims,
#                               blit=True)
#FFwriter = animation.FFMpegWriter(fps=fps,
#                                  metadata={'copyright': 'Public Domain'})
#
#anim.save('testingdiff.mp4', writer=FFwriter)
