import os.path

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import astropy.units as u
from astropy.constants import codata2018 as cst

from config.plotting import set_plotting_defaults, Linestyles, royalblue_palette, orangered_palette, save_figure
from config.settings import SHAPE_DIR
from config.units import Gauss
from src.electron_timescales import diffusion_timescale, synchrotron_timescale, inverse_compton_timescale


set_plotting_defaults()


def test_electron_timescales():
    # load experimental measurements and operate with them
    data = pd.read_csv(os.path.join(SHAPE_DIR, "shape_energies.csv"))
    E_min, E_max = (data["E_min"].to_numpy() * u.TeV).to(u.eV), (data["E_max"].to_numpy() * u.TeV).to(u.eV)
    dist, err = data["dist"].to_numpy() * u.pc, data["err"].to_numpy() * u.pc
    # t_min1 = diffusion_timescale(E_min, dist - err).value
    # t_min2 = diffusion_timescale(E_min, dist + err).value
    # t_max1 = diffusion_timescale(E_max, dist - err).value
    # t_max2 = diffusion_timescale(E_max, dist + err).value

    energies = np.logspace(11, 17, 200) * u.eV  # electron energies
    Bfield = 2**np.linspace(0, 2, 3) * 1e-6 * Gauss  # magnetic field values
    Dist = 50 * 2**np.linspace(0, 3, 4) * u.pc  # possible distances

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    left, right = 1e11, 1e17
    bottom, top = 1e2, 1e6

    # photon timescale
    ax.plot(1e15, 1e4, label='Photon time', linestyle='None')
    for i, D in enumerate(Dist[::-1]):
        photon_time = (D / cst.c).to(u.year) * np.ones_like(energies.value)
        plt.loglog(energies, photon_time, label=f"{D.value:.0f} pc",
                   color='slategray', alpha=0.8 - 0.15 * i, linestyle='solid', linewidth=2)
        plt.text(1.1 * energies[0].value, 1.2 * photon_time[0].value, f"{D.value:.0f} pc",
                 color='black', alpha=1 - 0.15 * i)

    # diffusion timescale measurements
    # for i, E in enumerate(zip(E_min.value, E_max.value)):
    #     plt.fill_between(E, (t_min1[i], t_max1[i]), (t_min2[i], t_max2[i]), alpha=.3)

    # diffusion timescale lines
    ax.plot(1e15, 1e4, label='Diffusion time', linestyle='None')
    for i, D in enumerate(Dist[::]):
        for j, b in enumerate(Bfield[::]):
            diffusion_time = diffusion_timescale(energies, D, b).to(u.year)
            if j == 0:
                plt.loglog(energies, diffusion_time, label=f"{D.value:.0f} pc, "r"$1~\mu$""G",
                           color=orangered_palette[i], alpha=0.8, linestyle=Linestyles[j % len(Linestyles)], linewidth=2)
                plt.text(1.1 * energies[0].value, 0.5 * diffusion_time[0].value, f"{D.value:.0f} pc",
                         rotation=-15, color=orangered_palette[i])
            else:
                plt.loglog(energies, diffusion_time,
                           color=orangered_palette[i], alpha=0.7, linestyle=Linestyles[j % len(Linestyles)], linewidth=2)

    # synchrotron timescale
    ax.plot(1e15, 1e4, label='Synchrotron time', linestyle='None')
    for i, B in enumerate(Bfield[::]):
        synch_time = synchrotron_timescale(energies, B, cst.m_e).to(u.year)
        plt.loglog(energies, synch_time, label=f"{B.value * 1e6:.2g} "r"$\mu G$",
                   color=royalblue_palette[i], alpha=.8, linestyle=Linestyles[i % len(Linestyles)], linewidth=2)
        top_i = np.argmin(np.abs(top - synch_time.value))
        plt.text(1.8 * energies[top_i].value, 0.45 * synch_time[top_i].value,
                 f"{B.value * 1e6:.2g} "r"$\mu G$", rotation=-40, color=royalblue_palette[i], ma='left')

    # inverse compton timescale
    ic_time = inverse_compton_timescale(energies, cst.m_e)
    ax.plot(1e15, 1e4, label='Inverse Compton', linestyle='None')
    ax.plot(energies, ic_time, label='time', color='seagreen', linestyle='solid', linewidth=2)
    ax.plot(1e15, 1e4, label='', linestyle='None')
    ax.plot(1e15, 1e4, label=' ', linestyle='None')
    ax.plot(1e15, 1e4, label=' ', linestyle='None')
    plt.text(6e15, 1e5, f"Inverse Compton", rotation=+30, color='seagreen', ma='left')

    # X-axis
    ax.set_xlabel("Electron energy, eV")
    ax.set_xlim(left, right)
    ax.xaxis.set_ticks_position('both')

    # Y-axis
    ax.set_ylabel("Timescale, years")
    ax.set_ylim(bottom, top)
    ax.yaxis.set_ticks_position('both')

    # both axes double ticks
    ax.tick_params(labelright=True, labeltop=True, top=True, right=True)

    # Grid
    ax.grid(linestyle='dashed', color='lightgray')
    plt.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.05),)

    fig.subplots_adjust(right=0.8)
    plt.tight_layout()

    save_figure("electron_timescales")
    plt.show()
    return


if __name__ == '__main__':
    test_electron_timescales()
