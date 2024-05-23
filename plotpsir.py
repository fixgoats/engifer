import os
from argparse import ArgumentParser, BooleanOptionalAction

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("--file")
parser.add_argument("--log", action=BooleanOptionalAction)
parser.add_argument("--psir", action=BooleanOptionalAction)
parser.add_argument("--psik", action=BooleanOptionalAction)
parser.add_argument("--psiksqrt", action=BooleanOptionalAction)
parser.add_argument("--normalise", action=BooleanOptionalAction)
parser.add_argument("--out")
args = parser.parse_args()

plots = []
a = np.load(args.file, allow_pickle=True).item()
if args.psir:
    psir = a["rpsidata"]
    extentr = a["extentr"]
    start = np.shape(psir)[0] // 6
    end = np.shape(psir)[0] - start
    extentr = [2 * x / 3 for x in extentr]
    fig, ax = plt.subplots()
    if args.log:
        im = ax.imshow(
            psir[start:end, start:end],
            origin="lower",
            interpolation="none",
            extent=extentr,
            norm=colors.LogNorm(vmin=psir.min(), vmax=psir.max()),
        )
        ax.set_title(r"$\log(|\psi_r|^2)$")
        ax.set_xlabel(r"x [µm]")
        ax.set_ylabel(r"y [µm]")
    else:
        im = ax.imshow(
            psir[start:end, start:end],
            origin="lower",
            interpolation="none",
            extent=extentr,
        )
        ax.set_title(r"$(|\psi_r|^2$")
        ax.set_xlabel(r"x [µm]")
        ax.set_ylabel(r"y [µm]")

    cb = plt.colorbar(im)
    if args.out is None:
        plt.show()

    else:
        if args.log:
            fig.savefig(os.path.join(args.out, "psirlog.pdf"))
            plots.append("psirlog.pdf")
        else:
            fig.savefig(os.path.join(args.out, "psir.pdf"))
            plots.append("psir.pdf")

if args.psik:
    psik = a["kpsidata"]
    if args.normalise:
        psik /= np.max(psik)
    extentk = a["extentk"]
    start = int(np.shape(psik)[0] / 2.5)
    end = np.shape(psik)[0] - start
    extentk = [x / 5 for x in extentk]
    fig, ax = plt.subplots()
    im = ax.imshow(
        psik[start:end, start:end], origin="lower", interpolation="none", extent=extentk
    )
    ax.set_xlabel(r"$k_x$ [µ$m^{-1}$]")
    ax.set_ylabel(r"$k_y$ [µ$m^{-1}$]")
    basename = "psik"
    if args.log:
        im.set_norm(colors.LogNorm(vmin=psik.min(), vmax=psik.max()))
        basename += "log"
        cb = plt.colorbar(im)
    if args.psiksqrt:
        im = ax.imshow(
            np.sqrt(psik[start:end, start:end]),
            origin="lower",
            interpolation="none",
            extent=extentk,
        )
        ax.set_title(r"$|\psi_k|^2$")
        ax.set_xlabel(r"$k_x$ [µ$m^{-1}$]")
        ax.set_ylabel(r"$k_y$ [µ$m^{-1}$]")
        plots.append("psiksqrt.pdf")
    else:
        im = ax.imshow(
            psik[start:end, start:end],
            origin="lower",
            interpolation="none",
            extent=extentk,
        )
        ax.set_title(r"$|\psi_k|^2$")
        ax.set_xlabel(r"$k_x$ [µ$m^{-1}$]")
        ax.set_ylabel(r"$k_y$ [µ$m^{-1}$]")
        plots.append("psik.pdf")

    if args.out is None:
        plt.show()

    else:
        if args.log:
            fig.savefig(os.path.join(args.out, "psiklog.pdf"))
            plots.append("psiklog.pdf")
        else:
            fig.savefig(os.path.join(args.out, "psik.pdf"))
            plots.append("psik.pdf")

print(f"made plots {plots}")
