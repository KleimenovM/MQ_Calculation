import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from config.settings import SPECTRUM_DIR, PICS_DIR
from config.units import flux_unit
from config.plotting import set_plotting_defaults, save_figure

dist = 6.6 * u.kpc  # [GAIA-2018]
area = 4 * np.pi * dist ** 2


def HAWC():
    f_u = u.TeV / (u.cm ** 2 * u.s)

    e, flux, f_p, f_l = np.loadtxt(os.path.join(SPECTRUM_DIR, "HAWC-one-asym.txt"), skiprows=1, unpack=True)

    e = (e * u.TeV).to(u.eV)
    flux = (flux * f_u).to(flux_unit)
    f_p, f_l = abs(f_p * f_u).to(flux_unit), abs(f_l * f_u).to(flux_unit)
    return e, flux, f_l, f_p, np.zeros_like(e), np.zeros_like(e)


def LHAASO():
    f_u = u.erg / (u.cm ** 2 * u.s)
    e, flux, f_l1, f_p1 = np.loadtxt(os.path.join(SPECTRUM_DIR, "LHAASO.txt"), skiprows=1, unpack=True)
    f_l, f_p = flux - f_l1, f_p1 - flux

    e = (e * u.TeV).to(u.eV)
    flux = (flux * f_u).to(flux_unit)
    f_p, f_l = (f_p * f_u).to(flux_unit), (f_l * f_u).to(flux_unit)
    return e, flux, f_l, f_p, np.zeros_like(e), np.zeros_like(e)


def HESS():
    e, f, f_p, f_n, e_n, e_p = np.loadtxt(os.path.join(SPECTRUM_DIR, "HESS-preliminary-reviewed.txt"), skiprows=1,
                                          unpack=True)

    e, e_n, e_p = (e * u.TeV).to(u.eV), (e_n * u.TeV).to(u.eV), (e_p * u.TeV).to(u.eV)
    f, f_p, f_n = f * flux_unit, f_p * flux_unit, f_n * flux_unit
    return e, f, f - f_n, f_p - f, e - e_n, e_p - e


def FERMI_Neronov():
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
    return e, flux, 0.3 * flux, .0 * np.ones_like(flux), de_low, de_high


def FERMI_Zhao():
    x = np.array([[1000, 3162.278183],
                  [3133.169682, 9816.754421],
                  [9816.754421, 30900.09571],
                  [31043.30149, 97264.03827],
                  [97714.72005, 301939.52]]) * u.MeV.to(u.eV)

    real_x = 10 ** (np.mean(np.log10(x), axis=1)) * u.eV
    x_err_right = x[:, 1] * u.eV - real_x
    x_err_left = real_x - x[:, 0] * u.eV

    y = np.array([[7.05E-13, 7.09E-13],
               [5.97E-13, 6.01E-13],
               [1.02E-12, 1.01E-12],
               [1.66E-12, 1.67E-12],
               [2.96E-12, 2.98E-12]]) * flux_unit

    real_y = np.mean(y, axis=1)
    return real_x, real_y, 0.3 * real_y, .0 * np.ones_like(real_y), x_err_left, x_err_right


def gamma_ray_data():
    names = ["HAWC, 2024", "LHAASO, 2024", "H.E.S.S., 2024",
             "FERMI, [Neronov, 2024]", "FERMI, [Zhao, 2025]"]
    functions = [HAWC, LHAASO, HESS, FERMI_Neronov, FERMI_Zhao]
    e, flux, f_l, f_p, e_l, e_p = [], [], [], [], [], []
    for i in range(len(names)):
        e_i, flux_i, f_l_i, f_p_i, e_l_i, e_p_i = functions[i]()
        e.append(e_i)
        flux.append(flux_i)
        f_l.append(f_l_i)
        f_p.append(f_p_i)
        e_l.append(e_l_i)
        e_p.append(e_p_i)
    return names, e, flux, f_l, f_p, e_l, e_p


def gamma_ray_measurements():
    names = ["HAWC, 2024", "LHAASO, 2024", "H.E.S.S., 2024",
             "FERMI, [Neronov, 2024]", "FERMI, [Zhao, 2025]"]
    functions = [HAWC, LHAASO, HESS, FERMI_Neronov, FERMI_Zhao]
    for i in range(len(names)):
        e, flux, f_l, f_p, e_l, e_p = functions[i]()
        plt.errorbar(e, flux, xerr=[e_l, e_p], yerr=[f_l, f_p], linestyle='None', label=names[i], fmt='.',
                     uplims=f_p <= 0)
    return


def plot_gamma_ray_measurements():
    set_plotting_defaults()
    gamma_ray_measurements()
    plt.xlabel("Energy [eV]")
    plt.xscale('log')

    plt.plot([1e10, 1e16], [3e-13, 3e-13])

    plt.ylabel("Flux [erg cm$^{-2}$ s$^{-1}$]")
    plt.yscale('log')
    plt.ylim(1e-13, 1e-10)

    plt.legend(loc=2)
    plt.tight_layout()
    save_figure("gamma_ray_measurements")
    plt.show()
    return


if __name__ == '__main__':
    plot_gamma_ray_measurements()
