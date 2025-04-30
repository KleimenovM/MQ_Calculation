import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import codata2010 as cst

from config.settings import SPECTRUM_DIR, PICS_DIR
from config.units import flux_unit
from config.constants import CST_HC

dist = 6.6 * u.kpc  # [GAIA-2018]
area = 4 * np.pi * dist ** 2


def optical_measurements():
    # --------
    # optical from data in the article
    name = "UKIRT (Optical), 1999"

    with open(os.path.join(SPECTRUM_DIR, "optical_1999.txt"), "r") as f:
        d = f.readlines()

    n = len(d) - 1
    energies, fluxes = np.zeros(n) * u.eV, np.zeros(n) * flux_unit
    for i, line in enumerate(d[1:]):
        dat = line.split()
        energies[i] = (CST_HC / (float(dat[0].strip()) * 0.1 * u.nm)).to(u.eV)
        fluxes[i] = abs(float(dat[1].strip()) * (u.W / (u.m ** 2 * u.um)) * CST_HC / energies[i]).to(flux_unit)

    plt.errorbar(energies.value, fluxes.value, fmt='.', label=name, linestyle='None')

    # --------
    # near infrared (NIR) measurements (plot from the article)
    name = "UKIRT (NIR), 1999"

    with open(os.path.join(SPECTRUM_DIR, "NIR_1999.txt"), "r") as f:
        d = f.readlines()

    n = len(d) - 1
    energies, fluxes = np.zeros(n) * u.eV, np.zeros(n) * flux_unit
    for i, line in enumerate(d[1:]):
        dat = line.split(',')
        energies[i] = (CST_HC / (float(dat[0].strip()) * u.um)).to(u.eV)
        fluxes[i] = abs(float(dat[1].strip()) * (u.W / (u.m ** 2 * u.um)) * CST_HC / energies[i]).to(flux_unit)

    plt.errorbar(energies.value, fluxes.value, yerr=0.1 * fluxes.value, fmt='.', label=name, linestyle='None')
    return