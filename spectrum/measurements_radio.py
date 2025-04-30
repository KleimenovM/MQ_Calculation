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


def radio_measurements():
    # --------
    # [radio-1999], flare measurements

    def get_data(freq, filename, name):
        path = os.path.join(SPECTRUM_DIR, filename)
        with open(path, 'r') as f:
            d = f.readlines()

        n = len(d) - 1

        result = np.zeros(n) * flux_unit
        e = ((freq * u.GHz) * cst.h).to(u.eV) * np.ones(n)

        for i, line in enumerate(d[1:]):
            result[i] = (float(line.strip()) * u.Jy / cst.h * e[i]).to(flux_unit)

        # alphas = np.linspace(1, 0.2, n)
        plt.errorbar(e.value, result.value, yerr=0.05 * result.value, linestyle='None', label=name, marker='.')
        return

    get_data(0.843, "radio_flare_1999_0.843GHz.txt", "1999, 0.843 GHz: MOST")
    get_data(1.4, "radio_flare_1999_1.4GHz.txt", "1999, 1.4 GHz: VLA, ATCA, MERLIN, RATAN")
    get_data(4.9, "radio_flare_1999_4.9GHz.txt", "1999, 4.9 GHz: VLA, ATCA, RATAN")
    get_data(8.4, "radio_flare_1999_8.4GHz.txt", "1999, 8.4 GHz: GBI, VLA, ATCA, RATAN")
    get_data(14.9, "radio_flare_1999_14.9GHz.txt", "1999, 14.9 GHz: VLA")

    # --------
    # [radio-2004]
    name = "2004, VLA"
    energies = (([1.425, 4.860, 8.460, 14.940, 22.460, 43.399] * u.GHz) * cst.h).to(u.eV)
    fluxes = ([12.50, 6.56, 4.45, 2.09, 1.96, 1.23] * u.mJy / cst.h * energies).to(flux_unit)
    errors = ([1.22, 0.31, 0.09, 0.18, 0.16, 0.31] * u.mJy / cst.h * energies).to(flux_unit)
    plt.errorbar(energies, fluxes, errors, fmt='.', label=name)

    # --------
    # [radio-2006]
    name = "2003-2005, VLA"
    energy = ([1280, 610, 235] * u.MHz * cst.h).to(u.eV)
    flux = ([49.41, 3.36, 6.27] * u.mJy / cst.h * energy).to(flux_unit)
    upper = [False, True, True]
    error = ([1.13, 1.13, 1.13] * u.mJy / cst.h * energy).to(flux_unit)
    plt.errorbar(energy, flux, error, fmt='.', label=name, uplims=upper)

    # --------
    # [keV-2011]
    name = "VLA, 2003"
    energy = (8.4 * u.GHz * cst.h).to(u.eV)
    lum = 6.3e26 * u.erg / u.s
    bandwidth = (100 * u.MHz * cst.h).to(u.eV)
    flux = (lum / area / bandwidth * energy).to(flux_unit)
    plt.errorbar(energy, flux, 0.2 * flux, fmt='.', label=name, uplims=True)
    return