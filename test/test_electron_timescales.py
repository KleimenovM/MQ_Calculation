import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

import astropy.units as u
from astropy.constants import codata2018 as cst

# Set default font size
mpl.rcParams['font.size'] = 14

# Grid parameters
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = 'dashed'
mpl.rcParams['grid.color'] = 'lightgray'

linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

from electron_timescales import diffusion_timescale

# load experimental measurements and operate with them
data = pd.read_csv("shape_energies.csv")
E_min, E_max = (data["E_min"].to_numpy() * u.TeV).to(u.eV), (data["E_max"].to_numpy() * u.TeV).to(u.eV)
dist, err = data["dist"].to_numpy() * u.pc, data["err"].to_numpy() * u.pc
t_min1 = diffusion_timescale(E_min, dist - err).value
t_min2 = diffusion_timescale(E_min, dist + err).value
t_max1 = diffusion_timescale(E_max, dist - err).value
t_max2 = diffusion_timescale(E_max, dist + err).value


from electron_timescales import synchrotron_timescale, Gauss, Franklin
from electron_timescales import inverse_compton_time

energies = np.logspace(11, 17, 400) * u.eV  # electron energies
Bfield = 2**np.linspace(0, 3, 4) * 1e-6 * Gauss  # magnetic field values
Dist = 50 * 2**np.linspace(0, 3, 4) * u.pc  # possible distances

fig = plt.figure(figsize=(12, 6))
ax = plt.gca()

left, right = 1e11, 1e17
bottom, top = 1e2, 1e6

# photon timescale
ax.plot(1e15, 1e4, label='Phot. time', linestyle='None')
for i, D in enumerate(Dist[::-1]):
    photon_time = (D / cst.c).to(u.year) * np.ones_like(energies.value)
    plt.loglog(energies, photon_time, label=f"{D.value:.0f} pc",
               color='black', alpha=0.8 - 0.15 * i, linestyle=linestyles[i % len(linestyles)], linewidth=2)
    plt.text(1.1 * energies[0].value, 1.2 * photon_time[0].value, f"{D.value:.0f} pc",
             color='black', alpha=1 - 0.15 * i)


# diffusion timescale measurements
for i, E in enumerate(zip(E_min.value, E_max.value)):
    plt.fill_between(E, (t_min1[i], t_max1[i]), (t_min2[i], t_max2[i]), alpha=.3)

# diffusion timescale lines
ax.plot(1e15, 1e4, label='Dif. time', linestyle='None')
for i, D in enumerate(Dist[::-1]):
    diffusion_time = diffusion_timescale(energies, D).to(u.year)
    plt.loglog(energies, diffusion_time, label=f"{D.value:.0f} pc",
               color='orangered', alpha=1 - 0.15 * i, linestyle=linestyles[i % len(linestyles)], linewidth=2)
    plt.text(1.1 * energies[0].value, 0.45 * diffusion_time[0].value, f"{D.value:.0f} pc",
             rotation=-15, color='orangered', alpha=1 - 0.12 * i)

# synchrotron timescale
ax.plot(1e15, 1e4, label='Synch. time', linestyle='None')
for i, B in enumerate(Bfield[::-1]):
    synch_time = synchrotron_timescale(energies, B, cst.m_e).to(u.year)
    plt.loglog(energies, synch_time, label=f"{B.value * 1e6:.2g} "r"$\mu G$",
               color='royalblue', alpha=1 - 0.15 * i, linestyle=linestyles[i % len(linestyles)], linewidth=2)
    top_i = np.argmin(np.abs(top - synch_time.value))
    plt.text(1.8 * energies[top_i].value, 0.4 * synch_time[top_i].value, f"{B.value * 1e6:.2g} "r"$\mu G$",
             rotation=-40, color='royalblue', alpha=1 - 0.1 * i, ma='left')

# inverse compton timescale
ic_time = inverse_compton_time(energies, cst.m_e)
ax.plot(1e15, 1e4, label=' ', linestyle='None')
ax.plot(energies, ic_time, label='IC time', color='green', linestyle='solid', linewidth=2)

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
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

fig.subplots_adjust(right=0.8)
plt.tight_layout()

plt.savefig('pictures/e_timescales.png', dpi=600)
plt.savefig('pictures/e_timescales.pdf')
plt.show()