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


def x_ray_measurements():
    # ------------------------------------
    # keV measurements [keV, 2015] figure 6
    name = "Swift/XRT + NuStar, 2014 (hard)"

    with open(os.path.join(SPECTRUM_DIR, "X-Ray.txt"), "r") as f:
        d = f.readlines()

    n = len(d) - 1
    energies, fluxes = np.zeros(n) * u.eV, np.zeros(n) * flux_unit
    for i, line in enumerate(d[1:]):
        dat = line.split(',')
        energies[i] = (float(dat[0].strip()) * u.keV).to(u.eV)
        fluxes[i] = abs(float(dat[1].strip()) * energies[i] / (u.cm ** 2 * u.s)).to(flux_unit)

    plt.errorbar(energies.value, fluxes.value, 0.1 * fluxes.value,
                 marker='o', markersize=3, color='#00d', alpha=.6, label=name, linestyle='None')

    # ------------------------------------
    # keV measurements [keV, 2015] figure 6
    name = "Swift/XRT + NuStar, 2014 (soft)"

    with open(os.path.join(SPECTRUM_DIR, "X-Ray-soft.txt"), "r") as f:
        d = f.readlines()

    n = len(d) - 1
    energies, counts = np.zeros(n) * u.eV, np.zeros(n) / (u.keV * u.s)
    for i, line in enumerate(d[1:]):
        dat = line.split(',')
        energies[i] = (float(dat[0].strip()) * u.keV).to(u.eV)
        counts[i] = abs(float(dat[1].strip()) / (u.keV * u.s))

    total_flux = 5.87 * 1e-10 * flux_unit

    normalization = np.trapezoid(counts, energies)

    fluxes = total_flux * energies * counts / normalization

    plt.errorbar(energies.value, fluxes.value, 0.1 * fluxes.value,
                 marker='^', markersize=3, color='#77f', alpha=.6, label=name, linestyle='None')

    # -------------------
    # nuSTAR measurements, 2021 [keV-2025] figure 7
    name = "NuStar, 2021"

    with open(os.path.join(SPECTRUM_DIR, "keV_spectrum-2025.txt"), "r") as f:
        d = f.readlines()

    n = len(d) - 1
    energies, fluxes = np.zeros(n) * u.eV, np.zeros(n) * flux_unit
    for i, line in enumerate(d[1:]):
        dat = line.split(',')
        energies[i] = (float(dat[0].strip()) * u.keV).to(u.eV)
        fluxes[i] = abs(float(dat[1].strip()) * u.erg / (u.cm ** 2 * u.s)).to(flux_unit)

    plt.errorbar(energies.value, fluxes.value, 0.2 * fluxes.value, fmt='.', label=name, linestyle='None')


    # --------------------------------
    # constraint, [Suzuki et al., 2025]

    name = "XRISM, 2024 (broad)"

    x = (np.array([2048.8939870443005, 10430.995959331496]) * u.eV).value
    y = (np.array([[5.226151097035918e-12, 5.281136737606013e-12],
                   [1.933586584317008e-12, 1.933586584317008e-12]]) * flux_unit).value

    plt.fill_between(x, y[0], y[1], label=name)

    # FOUR SPECTRA FROM REVNIVTSEV ET AL
    # [XRay-2002] Super-Eddington outburst
    names = ["RXTE/PCA/HEXTE, 1999, 50-170 s", "RXTE/PCA/HEXTE, 1999, 170-300 s",
             "RXTE/PCA/HEXTE, 1999, 500-700 s", "RXTE/PCA/HEXTE, 1999, 1100-1500 s"]
    filenames = ["keV_1999-I.txt", "keV_1999-II.txt", "keV_1999-III.txt", "keV_1999-IV.txt"]
    colors = ["#f00", "#f99", "#d00", "#f55"]
    markers = ["*", "o", "v", "s"]

    for i in range(len(names)):
        energy, flux = np.loadtxt(os.path.join(SPECTRUM_DIR, filenames[i]), skiprows=1, unpack=True, delimiter=',')

        energy = (energy * u.keV).to(u.eV)
        flux = (flux * u.keV ** 2 / (u.s * u.cm ** 2 * u.keV)).to(flux_unit)

        plt.scatter(energy, flux, marker=markers[i], s=15, label=names[i], color=colors[i], alpha=0.6)

    return