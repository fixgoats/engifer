from solvers import PeriodicSim1D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
def gauss(x):
    return np.exp(-x*x)

def sqr_mod(x):
    return x.real * x.real + x.imag * x.imag

samples = 1024
start = -10
end = 10
dx = (end-start)/samples
x = np.linspace(start, end, samples)
psi0 = gauss(x)
dt = 0.1
frames = 1200
fps = 60
t = np.arange(0, dt*frames, dt)
Ex = np.zeros(frames)
fftsim = PeriodicSim1D(psi0, start, end)
P = np.ones(frames)*65

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax2.set_ylim(4*start, 4*end)
ax3.set_ylim(65, 68)
line1, = ax1.plot(x, sqr_mod(psi0))
line2, = ax2.plot(t, Ex)
line3, = ax3.plot(t, P)
def animate(frame):
    fftsim.step(dt)
    line1.set_data(x, sqr_mod(fftsim.psi))
    Ex[frame] = np.sum(fftsim.psi.conjugate()*x*fftsim.psi)
    line2.set_data(t, Ex)
    P[frame] = np.sum(fftsim.psi.conjugate()*fftsim.psi)
    line3.set_data(t, P)
    return [line1, line2, line3]

anim = animation.FuncAnimation(fig, animate, frames=frames, blit=True)
FFwriter = animation.FFMpegWriter(fps=fps, metadata={'copyright': 'Public Domain'})
anim.save('1dfft.mp4', writer=FFwriter)
