import numpy as np
import torch
from src.solvers import SsfmGPCUDA, npnormSqr, tnormSqr, hbar
from src.sierpinski import getSierpinskiPoints
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from matplotlib import animation
import tomllib
import argparse
from datetime import date

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

datestamp = date.today()
parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--use-cached', action='store_true')
args = parser.parse_args()
if args.config is None and args.use_cached is None:
    exit('Need to specify config')

with open(f'{args.config}', 'rb') as f:
    pars = tomllib.load(f)


def gauss(x, y, sigmax, sigmay):
    return torch.exp(-x * x / sigmax - y * y / sigmay)


if args.use_cached is False:
    cuda = torch.device('cuda')
    endX = pars["endX"]
    startX = pars["startX"]
    samplesX = pars["samplesX"]
    endY = pars["endY"]
    startY = pars["startY"]
    samplesY = pars["samplesY"]
    dx = (endX - startX) / samplesX
    dy = (endY - startY) / samplesY
    x = torch.arange(startX, endX, dx)
    y = torch.arange(startY, endY, dy)
    x = x.type(dtype=torch.cfloat)
    y = y.type(dtype=torch.cfloat)
    gridY, gridX = torch.meshgrid(y, x, indexing='ij')
    kxmax = np.pi / dx
    kymax = np.pi / dy
    dkx = 2 * kxmax / samplesX
    psi = 0.1*torch.rand((samplesY, samplesX), dtype=torch.cfloat)

    # constV = ((gridX / 50)**2 + (gridY / 50)**2)**8 - 0.5j*pars["gammalp"]
    constV = -0.5j*pars["gammalp"]*torch.ones((samplesY, samplesX))

    points = np.array(getSierpinskiPoints(pars["divisions"]))
    points = points - (1+1j) * 2**(pars["divisions"] - 1)
    points = points * pars["radius"] * 0.5**(pars["divisions"])
    pump = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
    for p in points:
        pump += pars["pumpStrength"]*gauss(gridX - p.real,
                                           gridY - p.imag,
                                           pars["sigma"],
                                           pars["sigma"])

    nR = torch.zeros((samplesY, samplesX), dtype=torch.cfloat)
    gpsim = SsfmGPCUDA(dev=cuda,
                       psi0=psi,
                       gridX=gridX,
                       gridY=gridY,
                       m=pars["m"],
                       nR0=nR,
                       alpha=pars["alpha"],
                       Gamma=pars["Gamma"],
                       gammalp=pars["gammalp"],
                       R=pars["R"],
                       pump=pump,
                       G=pars["G"],
                       eta=pars["eta"],
                       constV=constV,
                       dt=pars["dt"])

    nframes = 1024
    fps = 24
    fig, [ax1, ax2] = plt.subplots(1, 2)
    fig.dpi = 300
    fig.figsize = (6.4, 3.6)
    extentr = [startX, endX, startY, endY]
    extentk = [-kxmax/2, kxmax/2, -kymax/2, kymax/2]
    im1 = ax1.imshow(npnormSqr(gpsim.psi.cpu().detach().numpy()),
                     origin='lower',
                     extent=extentr)
    ax1.set_title(r'$|\psi_r|^2, t = 0 ps$')
    psik0 = npnormSqr(gpsim.psik.cpu().detach().numpy())
    im2 = ax2.imshow(np.log(fftshift(psik0) + 1)[255:samplesY-256, 255:samplesX-256],
                     origin='lower',
                     extent=extentk)
    ax2.set_title(r'$\ln(|\psi_k|^2 + 1)$')

    positions = ax1.scatter(points.real,
                            points.imag,
                            s=0.5,
                            linewidths=0.1,
                            color='#ff6347')
    ax1.set_xlabel(r'x ($\mu$m)')
    ax1.set_ylabel(r'y ($\mu$m)')
    # fraction=0.046 and pad=0.04 are magic settings that just work for some reason
    # to make the colorbar the same height as the graph.
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    dispersion = np.zeros((nframes, samplesX), dtype=complex)

    for i in range(nframes):
        gpsim.step()
        rdata = gpsim.psi.detach().cpu().numpy()
        dispersion[i, :] = rdata[samplesY//2 - 1, :]

    # def init():
    #     return [im1, im2]
    # 
    # 
    # def animate_heatmap(frame):
    #     gpsim.step()
    #     rdata = gpsim.psi.detach().cpu().numpy()
    #     kdata = gpsim.psik.detach().cpu().numpy()
    #     bleh[frame, :] = rdata[samplesY//2 - 1, :]
    #     kdata = np.log(npnormSqr(fftshift(kdata))[255:samplesY-256, 255:samplesX-255].real + 1)
    #     im1.set_data(rdata)
    #     im2.set_data(kdata)
    #     vmin = np.min(kdata)
    #     vmax = np.max(kdata)
    #     im2.set_clim(vmin, vmax)
    #     vmin = np.min(rdata)
    #     vmax = np.max(rdata)
    #     im1.set_clim(vmin, vmax)
    #     ax1.set_title(f'$|\\psi_r|^2$, t = {gpsim.t:.2f}')
    #     return [im1, im2]
    # 
    # 
    # anim = animation.FuncAnimation(fig,
    #                                animate_heatmap,
    #                                init_func=init,
    #                                frames=nframes,
    #                                blit=True)
    # FFwriter = animation.FFMpegWriter(fps=fps, metadata=pars)
    # 
    # path = f'animations/penroser{pars["radius"]}'\
    #        f'd{pars["divisions"]}'\
    #        f'p{pars["pumpStrength"]}'\
    #        f'n{pars["samplesX"]}'\
    #        f's{pars["sigma"]}.mp4'
    # 
    # anim.save(path, writer=FFwriter)

    rdata = gpsim.psi.detach().cpu().numpy()
    np.save("rdata.npy", rdata)
    kdata = gpsim.psik.detach().cpu().numpy()
    np.save("kdata.npy", kdata)
    np.save("disp.npy", dispersion)
    np.save("extentr.npy", extentr)
    np.save("extentk.npy", extentk)
    Emax = hbar * np.pi / pars['dt']
    extentE = [-kxmax, kxmax, 0, Emax]
    np.save("extentE.npy", extentE)

    plt.cla()
    fig, ax = plt.subplots()
    im = ax.imshow(npnormSqr(rdata), origin='lower',
                   extent=extentr)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = f'graphs/{datestamp}/'\
           f'sierpinskirr{pars["radius"]}'\
           f'd{pars["divisions"]}'\
           f'p{pars["pumpStrength"]}'\
           f'n{pars["samplesX"]}'\
           f's{pars["sigma"]}.pdf'
    ax.set_title(r'$|\psi_r|^2$')
    ax.set_xlabel(r'x ($\mu$m)')
    ax.set_ylabel(r'y ($\mu$m)')
    plt.savefig(path)

    plt.cla()
    fig, ax = plt.subplots()

    kdata = npnormSqr(fftshift(kdata)[255:samplesY-256, 255:samplesX-256])
    im = ax.imshow(np.log(kdata+np.exp(-10)), origin='lower',
                   extent=extentk)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = f'graphs/{datestamp}/'\
           f'sierpinskikr{pars["radius"]}'\
           f'd{pars["divisions"]}'\
           f'p{pars["pumpStrength"]}'\
           f'n{pars["samplesX"]}'\
           f's{pars["sigma"]}.pdf'
    ax.set_title(r'$\ln(|\psi_k|^2 + e^{-10})$')
    ax.set_xlabel(r'k_x ($\hbar/\mu$ m)')
    ax.set_ylabel(r'k_y ($\hbar/\mu$ m)')
    plt.savefig(path)

    plt.cla()
    dispersion = fftshift(fft(ifft(dispersion, axis=0), axis=1))
    if not pars["wneg"]:
        start = dispersion.shape[0] // 2 - 1
        dispersion = dispersion[start:, :]
    fig, ax = plt.subplots()
    im = ax.imshow(np.log(np.sqrt(npnormSqr(dispersion))),
                   aspect='auto',
                   origin='lower',
                   extent=extentE)
    path = f'graphs/{datestamp}/'\
           f'sierpinskidispersionr{pars["radius"]}'\
           f'd{pars["divisions"]}'\
           f'p{pars["pumpStrength"]}'\
           f'n{pars["samplesX"]}'\
           f's{pars["sigma"]}.pdf'
    ax.set_title('$E(k_x)$, logarithmic intensity')
    ax.set_xlabel(r'$k_x$ ($\mu m^{-1}$)')
    ax.set_ylabel(r'$E$ (meV)')
    plt.savefig(path)

else:
    rdata = np.load("rdata.npy")
    kdata = np.load("kdata.npy")
    dispersion = np.load("disp.npy")
    extentr = np.load("extentr.npy")
    extentk = np.load("extentk.npy")
    extentE = np.load("extentE.npy")
    fig, ax = plt.subplots()
    im = ax.imshow(npnormSqr(rdata), origin='lower',
                   extent=extentr)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = f'graphs/{datestamp}/'\
           f'sierpinskirr{pars["radius"]}'\
           f'd{pars["divisions"]}'\
           f'p{pars["pumpStrength"]}'\
           f'n{pars["samplesX"]}'\
           f's{pars["sigma"]}.pdf'
    ax.set_title(r'$|\psi_r|^2$')
    ax.set_xlabel(r'x ($\mu m$)')
    ax.set_ylabel(r'y ($\mu m$)')
    plt.savefig(path)

    plt.cla()
    fig, ax = plt.subplots()

    samplesY = kdata.shape[0]
    beginkY = samplesY//4-1
    endkY = samplesY-samplesY//4
    samplesX = kdata.shape[1]
    beginkX = samplesX//4 - 1
    endkX = samplesX-samplesX//4
    kdata = npnormSqr(fftshift(kdata)[beginkY:endkY, beginkX:endkX])
    im = ax.imshow(np.log(kdata+np.exp(-10)), origin='lower',
                   extent=extentk)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = f'graphs/{datestamp}/'\
           f'sierpinskikr{pars["radius"]}'\
           f'd{pars["divisions"]}'\
           f'p{pars["pumpStrength"]}'\
           f'n{pars["samplesX"]}'\
           f's{pars["sigma"]}.pdf'
    ax.set_title(r'$\ln(|\psi_k|^2 + e^{-10})$')
    ax.set_xlabel(r'k_x ($\mu m^{-1}$)')
    ax.set_ylabel(r'$k_y$ ($\mu m^{-1}$)')
    plt.savefig(path)

    plt.cla()
    dispersion = fftshift(fft(ifft(dispersion, axis=0), axis=1))
    if not pars["wneg"]:
        start = dispersion.shape[0] // 2 - 1
        dispersion = dispersion[start:, :]
    fig, ax = plt.subplots()
    im = ax.imshow(np.log(np.sqrt(npnormSqr(dispersion))),
                   aspect='auto',
                   origin='lower',
                   extent=extentE)
    path = f'graphs/{datestamp}/'\
           f'sierpinskidispersionr{pars["radius"]}'\
           f'd{pars["divisions"]}'\
           f'p{pars["pumpStrength"]}'\
           f'n{pars["samplesX"]}'\
           f's{pars["sigma"]}.pdf'
    ax.set_title('$E(k_x)$, logarithmic intensity')
    ax.set_xlabel(r'$k_x$ ($\mu m^{-1}$)')
    ax.set_ylabel(r'$E$ (meV)')
    plt.savefig(path)
