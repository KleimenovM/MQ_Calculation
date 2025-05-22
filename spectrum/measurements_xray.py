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


def constraint_XRISM():
    flux_per_area_unit = flux_unit / u.arcmin**2
    x = np.array([2e3, 1e4]) * u.eV  # 2-10 keV band
    x_mid = np.array([np.sqrt(x[0] * x[1]).value]) * u.eV
    x_err_l, x_err_p = x_mid - x[0], x[1] - x_mid
    y_per_arcmin = 9.2e-15 * flux_per_area_unit
    y_err = 1.1e-15 * flux_per_area_unit

    # article area value 629.4 arcmin2
    emission_area_min = 1 * u.deg * 0.2 * u.deg
    emission_area_max = 4 * emission_area_min
    emission_area = np.sqrt(emission_area_min * emission_area_max)

    y_real = np.array([(y_per_arcmin * emission_area).to(flux_unit).value]) * flux_unit
    y_err_min = y_real - ((y_per_arcmin - y_err) * emission_area_min).to(flux_unit)
    y_err_max = ((y_per_arcmin + y_err) * emission_area_max).to(flux_unit) - y_real
    return x_mid, y_real, y_err_min, y_err_max, x_err_l, x_err_p


def constraint_Chandra():
    x = (np.array([3.3513512672260234e-9]) * u.TeV).to(u.eV)
    x_low, x_high = (np.array([1.2248132360494609e-9, 9.532031470019544e-9]) * u.TeV).to(u.eV)
    y = np.array([1.1094676214563854e-11]) * flux_unit
    y_err = np.zeros(1) * flux_unit
    return x, y, y_err, y_err, x - x_low, x_high - x


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
    names = ["Swift/XRT + NuStar, 2014 (hard)", "NuStar, 2021", "XRISM, 2024 (broad)", "Chandra, 2024 (H.E.S.S.)"]
    functions = [hard_state_spectrum, nu_star_2021, constraint_XRISM, constraint_Chandra]

    for i in range(len(names)):
        print(names[i])
        e, f, f_l, f_p, e_l, e_p = functions[i]()
        plt.errorbar(e, f, xerr=[e_l, e_p], yerr=[f_l, f_p],
                     marker='o', markersize=3, label=names[i], linestyle='None')

    revnivtsev_spectra()

    return
