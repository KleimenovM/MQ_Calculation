import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

from config.plotting import Tab10, set_plotting_defaults
from config.settings import SPECTRUM_DIR, ELECTRONS_DIR
from src.electron_spectrum_parametrization import SpectrumParametrization
from tabulate_ic_cross_section import load_tabulated_matrix


def test_tabulated_matrix():
    electron_energy, photon_energy, result = load_tabulated_matrix()

    xx, yy = np.meshgrid(electron_energy.value, photon_energy.value, indexing='ij')
    plt.pcolormesh(xx, yy, np.log10(result.value + np.finfo(float).tiny), vmin=-36)
    plt.colorbar()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    return


def test_interpolated_matrix():
    data = pickle.load(open(os.path.join(ELECTRONS_DIR, "spectrum_interpolated.pck"), "rb"))
    x, y, f = data[0], data[1], data[2]

    xx, yy = np.meshgrid(x, y, indexing='ij')
    plt.pcolormesh(xx, yy, f((xx, yy)), vmin=-36)
    plt.show()
    return


def test_tabulated_spectrum():
    electron_energy, photon_energy, result = load_tabulated_matrix()

    spec = SpectrumParametrization(n0=1.0 / (u.eV * u.cm ** 3), e0=1e12 * u.eV,
                                   eta0=0.0, p0=1.6, k10=0.0, k20=2.6)

    data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
    names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data

    f_cor_long = []
    for f in f_cor:
        for f_j in f:
            f_cor_long.append(f_j)

    spec_vals = spec.dn_de0(electron_energy)
    photon_spectrum = trapezoid(spec_vals * result.T, electron_energy, axis=1)

    value = max(f_cor_long) / max(photon_energy**2 * photon_spectrum)
    print(value)

    set_plotting_defaults()
    plt.loglog(photon_energy, 0.2 * photon_energy**2 * photon_spectrum * value, color='black', linestyle='--')

    for i, name in enumerate(names):
        plt.errorbar(e[i], f_cor[i], xerr=[e_l[i], e_p[i]], yerr=[f_l_cor[i], f_p_cor[i]],
                     fmt='o', linestyle='None', uplims=f_p_cor[i] <= 0,
                     color=Tab10[i], label=f'{names[i]}')

    plt.xlim(1e8, 1e16)
    plt.xlabel("Energy, eV")

    plt.ylim(1e-14, 1e-10)
    plt.ylabel(r"Flux, $\mathrm{erg~cm^{-2}~s^{-1}}$")

    plt.tight_layout()
    plt.show()
    return


def test_interpolated_spectrum():
    data = pickle.load(open(os.path.join(ELECTRONS_DIR, "spectrum_interpolated.pck"), "rb"))
    x, y, f = data[0][::10], data[1][::10], data[2]

    xy, yx = np.meshgrid(x, y, indexing='ij')

    electron_energy = 10**x * u.eV
    photon_energy = 10**y * u.eV

    matrix = 10**f((xy, yx)) / (u.eV * u.s)

    spec = SpectrumParametrization(n0=1.0 / (u.eV * u.cm ** 3), e0=1e12 * u.eV,
                                   eta0=0.0, p0=1.6, k10=0.0, k20=2.6)

    data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
    names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data

    f_cor_long = []
    for f in f_cor:
        for f_j in f:
            f_cor_long.append(f_j)

    spec_vals = spec.dn_de0(electron_energy)
    photon_spectrum = trapezoid(spec_vals * matrix.T, electron_energy, axis=1)

    value = max(f_cor_long) / max(photon_energy ** 2 * photon_spectrum)
    print(value)

    set_plotting_defaults()
    plt.loglog(photon_energy, 0.2 * photon_energy ** 2 * photon_spectrum * value, color='black', linestyle='--')

    for i, name in enumerate(names):
        plt.errorbar(e[i], f_cor[i], xerr=[e_l[i], e_p[i]], yerr=[f_l_cor[i], f_p_cor[i]],
                     fmt='o', linestyle='None', uplims=f_p_cor[i] <= 0,
                     color=Tab10[i], label=f'{names[i]}')

    plt.xlim(1e8, 1e16)
    plt.xlabel("Energy, eV")

    plt.ylim(1e-14, 1e-10)
    plt.ylabel(r"Flux, $\mathrm{erg~cm^{-2}~s^{-1}}$")

    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    # test_tabulated_matrix()
    # test_interpolated_matrix()
    test_tabulated_spectrum()
    # test_interpolated_spectrum()
