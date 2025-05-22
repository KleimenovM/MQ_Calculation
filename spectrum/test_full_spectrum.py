import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import codata2010 as cst

from config.plotting import set_plotting_defaults, save_figure
from config.settings import SPECTRUM_DIR, PICS_DIR
from config.units import flux_unit

from spectrum.measurements_radio import radio_measurements
from spectrum.measurements_xray import x_ray_measurements
from spectrum.measurements_optical import optical_measurements
from spectrum.measurements_gamma import gamma_ray_measurements

dist = 6.1 * u.kpc  # [GAIA-2018]
area = 4 * np.pi * dist ** 2


def fit_low_energy_part():
    e = np.logspace(-8, 9, 1000) * u.eV
    e_peak = 7e4 * u.eV
    x = e / e_peak
    A = 1e-12 * flux_unit / u.eV
    shape = A * (x / 2) ** (-2 / 7) * np.exp(-x)
    plt.plot(e, e * shape, alpha=.5, linestyle='dashed', color='black')

    print(f"Total luminocity in excited state: {(np.trapezoid(shape, e).to(flux_unit) * area).to(u.erg / u.s):.2g}")

    ### Non-excited state
    e = np.logspace(-8, 9, 1000) * u.eV
    e_peak = 5e3 * u.eV
    x = e / e_peak
    A = 5e-14 * flux_unit / u.eV
    shape = A * (x / 2) ** (0) * np.exp(-x)
    plt.plot(e, e * shape, alpha=.5, linestyle='dashed', color='black')

    print(f"Total luminocity in passive state: {(np.trapezoid(shape, e).to(flux_unit) * area).to(u.erg / u.s):.2g}")
    return


def test_full_spectrum():
    set_plotting_defaults()

    plt.figure(figsize=(12, 7))

    radio_measurements()
    optical_measurements()
    x_ray_measurements()
    gamma_ray_measurements()

    fit_low_energy_part()

    # plot settings
    ax = plt.gca()

    ax.set_xscale('log')
    ax.set_xlabel(r"Energy, eV")

    ax.set_ylabel(r"$\epsilon^2 dN/d\epsilon,~\mathrm{erg~cm^{-2}~s^{-1}}$")
    ax.set_yscale('log')
    plt.ylim(1e-18, 1e-7)
    plt.xlim(1e-8, 1e16)

    plt.grid(color='lightgray', linestyle='--')
    plt.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.05),)

    plt.tight_layout()
    plt.tight_layout()
    # save_figure("spectrum")
    plt.show()
    return


if __name__ == '__main__':
    test_full_spectrum()
