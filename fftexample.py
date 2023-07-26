from solvers import PeriodicSim, PeriodicSim1D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

def gauss(x, y):
    return np.exp(-x*x - y*y)

def sqr_mod(x):
    return x.real*x.real + x.imag*x.imag

startX = -5
endX = 5
startY = -5
endY = 5
samples = 128
dx = (endX - startX)/samples
x = np.arange(startX, endX, dx)
xv, yv = np.meshgrid(x, x)
psi0 = gauss(xv, yv)#*(1-(xv/5)**6*(yv/5)**6)

fftsim = PeriodicSim(psi0, startX, endX, startY, endY)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xv, yv, sqr_mod(fftsim.psi), cmap=cm.viridis)

def animate(frame):
    ax.clear()
    fftsim.step(0.2)
    surf = ax.plot_surface(xv, yv, sqr_mod(fftsim.psi), cmap=cm.viridis)
    surf._facecolors2d=surf._facecolor3d
    surf._edgecolors2d=surf._edgecolor3d
    return surf,

anim = animation.FuncAnimation(fig, animate, frames=600, blit=True)
FFwriter = animation.FFMpegWriter(fps=30, metadata={'copyright': 'Public Domain'})

anim.save('testingfft.mp4', writer=FFwriter)
