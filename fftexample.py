from solvers import PeriodicSim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
plt.xlim(-5, 5)
plt.ylim(-5, 5)

def gauss(x, y):
    return np.exp(-x*x - y*y)

def sqr_mod(x):
    return x.real*x.real + x.imag*x.imag

x = np.linspace(-2, 2, 32)
xv, yv = np.meshgrid(x, x)
psi0 = gauss(xv, yv)#*(1-(xv/5)**6*(yv/5)**6)

fftsim = PeriodicSim(psi0, -10, 10, -10, 10)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_zlim(0, 1)
def animate(frame):
    ax.clear()
    fftsim.step(0.2)
    surf = ax.plot_surface(xv, yv, sqr_mod(fftsim.psi), cmap=cm.viridis)
    return surf,

anim = animation.FuncAnimation(fig, animate, frames=600, blit=True)
FFwriter = animation.FFMpegWriter(fps=30, metadata={'copyright': 'Public Domain'})

anim.save('testingfft.mp4', writer=FFwriter)
