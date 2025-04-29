import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import codata2010 as cst

from config.settings import SPECTRUM_DIR, PICS_DIR
from config.units import flux_unit, cst_hc

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
        energies[i] = (cst_hc / (float(dat[0].strip()) * 0.1 * u.nm)).to(u.eV)
        fluxes[i] = abs(float(dat[1].strip()) * (u.W / (u.m ** 2 * u.um)) * cst_hc / energies[i]).to(flux_unit)

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
        energies[i] = (cst_hc / (float(dat[0].strip()) * u.um)).to(u.eV)
        fluxes[i] = abs(float(dat[1].strip()) * (u.W / (u.m ** 2 * u.um)) * cst_hc / energies[i]).to(flux_unit)

    plt.errorbar(energies.value, fluxes.value, yerr=0.1 * fluxes.value, fmt='.', label=name, linestyle='None')
    return


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


def gamma_ray_measurements():
    # --------
    # [HAWC]
    name = "HAWC, 2024"

    f_u = u.TeV / (u.cm ** 2 * u.s)

    e, flux, f_p, f_l = np.loadtxt(os.path.join(SPECTRUM_DIR, "HAWC-one-asym.txt"), skiprows=1, unpack=True)

    e = (e * u.TeV).to(u.eV)
    flux = (flux * f_u).to(flux_unit)
    f_p, f_l = abs(f_p * f_u).to(flux_unit), abs(f_l * f_u).to(flux_unit)

    plt.errorbar(e, flux, yerr=[f_l, f_p], linestyle='None', label=name, fmt='.', uplims=f_p <= 0)

    # --------
    # [LHAASO]
    name = "LHAASO, 2024"

    f_u = u.TeV / (u.cm ** 2 * u.s)

    e, flux, f_p, f_l = np.loadtxt(os.path.join(SPECTRUM_DIR, "LHAASO.txt"), skiprows=1, unpack=True)

    e = (e * u.TeV).to(u.eV)
    flux = (flux * f_u).to(flux_unit)
    f_p, f_l = (f_p * f_u).to(flux_unit), (f_l * f_u).to(flux_unit)

    plt.errorbar(e, flux, yerr=[f_l, f_p], linestyle='None', label=name, fmt='.')

    # --------
    # [H.E.S.S.]
    name = "H.E.S.S., 2024"
    e, f, f_p, f_n, e_n, e_p = np.loadtxt(os.path.join(SPECTRUM_DIR, "HESS-preliminary-reviewed.txt"), skiprows=1, unpack=True)

    e, e_n, e_p = (e * u.TeV).to(u.eV), (e_n * u.TeV).to(u.eV), (e_p * u.TeV).to(u.eV)
    f, f_p, f_n = f * flux_unit, f_p * flux_unit, f_n * flux_unit

    plt.errorbar(e, f, xerr=[e - e_n, e_p - e], yerr=[f - f_n, f_p - f], linestyle='None', label=name, fmt='.')

    # --------
    # [FERMI] (upper limits, by A. Neronov)
    name = "FERMI, 2024"
    with open(os.path.join(SPECTRUM_DIR, "fermi.txt"), 'r') as f:
        d = f.readlines()

    n = len(d)
    dat = np.zeros([n - 1, 5])
    for i in range(1, n):
        line = d[i].split()
        for j in range(4):
            dat[i - 1, j] = float(line[j].strip())

    e = (dat[:, 2] * u.MeV).to(u.eV)
    flux = dat[:, 3] * flux_unit
    e_low, e_high = (dat[:, 0] * u.MeV).to(u.eV), (dat[:, 1] * u.MeV).to(u.eV)
    de_low, de_high = e - e_low, e_high - e

    plt.errorbar(e, flux, xerr=[de_low, de_high], yerr=.3 * flux, linestyle='None', label=name, fmt='.', uplims=True)

    # --------
    # [PLANCK] (upper limits)
    name = "Planck / WISE, 2008-2023"

    x = np.array([[1004.4166869027495, 3089.95530556224],
                  [3229.1695064433593, 9846.943932531487],
                  [9760.535077688362, 30159.617148429475],
                  [30831.528479163662, 95268.12507961658],
                  [96962.22330955854, 300932.3416927414]]) * u.MeV.to(u.eV)

    real_x = 10 ** (np.mean(np.log10(x), axis=1))
    x_err_right = x[:, 1] - real_x
    x_err_left = real_x - x[:, 0]

    y = np.array([[1.6665488096621727e-13, 1.6116569239501234e-13],
                  [1.4946704426036537e-13, 1.5585736598315709e-13],
                  [5.160921161399877e-13, 4.990933524123363e-13],
                  [5.611654314295999e-13, 5.658838112915587e-13],
                  [1.7820053488037003e-12, 1.7524122627745403e-12]]) * u.erg

    real_y = np.mean(y, axis=1)

    plt.errorbar(real_x, real_y, xerr=[x_err_left, x_err_right], yerr=0.5 * real_y, uplims=True, label=name, fmt='.')
    return


def fit_low_energy_part():
    e = np.logspace(-8, 9, 1000) * u.eV
    e_peak = 7e4 * u.eV
    x = e / e_peak
    A = 1e-12 * flux_unit / u.eV
    shape = A * (x / 2) ** (-2 / 7) * np.exp(-x)
    plt.plot(e, e * shape, alpha=.5, linestyle='dashed', color='black')

    print(f"Total luminocity in excited state: {(np.trapezoid(shape, e).to(flux_unit) * area).to(u.erg / u.s):.2g}")
    # %% md

    ### Non-excited state
    # %%
    e = np.logspace(-8, 9, 1000) * u.eV
    e_peak = 5e3 * u.eV
    x = e / e_peak
    A = 5e-14 * flux_unit / u.eV
    shape = A * (x / 2) ** (0) * np.exp(-x)
    plt.plot(e, e * shape, alpha=.5, linestyle='dashed', color='black')

    print(f"Total luminocity in passive state: {(np.trapezoid(shape, e).to(flux_unit) * area).to(u.erg / u.s):.2g}")
    return


def test_full_spectrum():
    plt.figure(figsize=(12, 8))

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
    plt.ylim(1e-21, 1e-7)
    plt.xlim(1e-8, 1e16)

    plt.grid(color='lightgray', linestyle='--')
    plt.legend(ncol=3)

    plt.tight_layout()
    plt.savefig(os.path.join(PICS_DIR, "spectrum.png"), dpi=600)
    plt.savefig(os.path.join(PICS_DIR, "spectrum.pdf"))
    plt.show()
    return


if __name__ == '__main__':
    test_full_spectrum()
