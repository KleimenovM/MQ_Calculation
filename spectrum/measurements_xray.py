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


def hard_state_spectrum():
    # keV measurements [keV, 2015] figure 6
    energies, fluxes = np.loadtxt(os.path.join(SPECTRUM_DIR, "X-Ray-hard.txt"), skiprows=1, unpack=True, delimiter=',')

    energies = (energies * u.keV).to(u.eV)
    fluxes = (fluxes * 1 / (u.cm ** 2 * u.s) * energies).to(flux_unit)
    f_err = np.zeros_like(fluxes)
    e_err = np.zeros_like(energies)
    return energies, fluxes, f_err, f_err, e_err, e_err


def nu_star_2021():
    # nuSTAR measurements, 2021 [keV-2025] figure 7

    energies, fluxes = np.loadtxt(os.path.join(SPECTRUM_DIR, "keV_spectrum-2025.txt"),
                                  skiprows=1, unpack=True, delimiter=',')

    energies = (energies * u.keV).to(u.eV)
    fluxes = fluxes * flux_unit

    f_err = np.zeros_like(fluxes)
    e_err = np.zeros_like(energies)

    return energies, fluxes, f_err, f_err, e_err, e_err


def constraint_2025_Suzuki():
    x = np.array([2048.8939870443005, 10430.995959331496])
    y = np.array([[5.226151097035918e-12, 5.281136737606013e-12],
                [1.933586584317008e-12, 1.933586584317008e-12]])

    energy = np.array([np.sqrt(x[0] * x[1])]) * u.eV
    flux = np.array([np.prod(y, axis=(0, 1))**(1/4)]) * flux_unit
    f_l, f_p = np.sqrt(np.prod(y[0])) * flux_unit - flux, flux - np.sqrt(np.prod(y[1])) * flux_unit
    e_l, e_p = energy - x[0] * u.eV, x[1] * u.eV - energy
    print(e_l, e_p, f_l, f_p)
    return energy, flux, f_l, f_p, e_l, e_p


def revnivtsev_spectra():
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


def x_ray_measurements():
    names = ["Swift/XRT + NuStar, 2014 (hard)", "NuStar, 2021", "XRISM, 2024 (broad)"]
    functions = [hard_state_spectrum, nu_star_2021, constraint_2025_Suzuki]

    for i in range(len(names)):
        print(names[i])
        e, f, f_l, f_p, e_l, e_p = functions[i]()
        plt.errorbar(e, f, xerr=[e_l, e_p], yerr=[f_l, f_p],
                     marker='o', markersize=3, label=names[i], linestyle='None')

    revnivtsev_spectra()

    return
