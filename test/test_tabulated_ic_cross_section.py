import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.integrate import trapezoid

from scipy.interpolate import splrep, splev

from config.plotting import Tab10, set_plotting_defaults, save_figure
from config.settings import SPECTRUM_DIR, INVERSE_COMPTON_DIR
from src.electron_spectrum_parametrization import SpectrumParametrization
from precomputing.tabulate_ic_cross_section import load_tabulated_matrix


def test_tabulated_matrix(filename):
    electron_energy, photon_energy, result = load_tabulated_matrix(filename)

    xx, yy = np.meshgrid(electron_energy, photon_energy, indexing='ij')
    plt.pcolormesh(xx, yy, np.log10(result + np.finfo(float).tiny), vmin=-36)
    plt.colorbar()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    return


def test_interpolated_matrix(filename):
    data = pickle.load(open(os.path.join(INVERSE_COMPTON_DIR, filename), "rb"))
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


def test_interpolated_spectrum(filename, color='black', linestyle='dashed', label=None,
                               if_show=False):
    data = pickle.load(open(os.path.join(INVERSE_COMPTON_DIR, filename), "rb"))
    x, y, f = data[0], data[1], data[2]

    xy, yx = np.meshgrid(x, y, indexing='ij')

    electron_energy = 10 ** x * u.eV
    photon_energy = 10 ** y * u.eV

    matrix = 10 ** f((xy, yx)) / (u.eV * u.s)

    spec = SpectrumParametrization(n0=1.0 / (u.eV * u.cm ** 3), e0=1e12 * u.eV,
                                   eta0=0.5, p0=1.9, k10=-10.0, k20=3.2)

    data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
    names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data
    names_short = ["HAWC", "LHAASO", "H.E.S.S.", "FERMI [Semikoz et al.]", "FERMI [Zhao et al.]"]

    f_cor_long = []
    for f in f_cor:
        for f_j in f:
            f_cor_long.append(f_j)

    spec_vals = spec.dn_de0(electron_energy)
    photon_spectrum = trapezoid(spec_vals * matrix.T, electron_energy, axis=1)

    resulting_flux = photon_energy ** 2 * photon_spectrum * 3e-25
    lg_phot_e = np.log10(photon_energy.value)
    spl = splrep(x=lg_phot_e, y=np.log10(resulting_flux.value), s=10)
    flux = splev(lg_phot_e, spl)
    plt.loglog(photon_energy, 10 ** flux, color=color, linestyle=linestyle, label=label)

    if if_show:
        for i, name in enumerate(names):
            plt.errorbar(e[i], f_cor[i], xerr=[e_l[i], e_p[i]], yerr=[f_l_cor[i], f_p_cor[i]],
                         fmt='o', linestyle='None', uplims=f_p_cor[i] <= 0,
                         color=Tab10[i], label=f'{names_short[i]}', capsize=2)
        plt.errorbar(1, 1, linewidth=0, linestyle=None, marker=None, label=' ')

        plt.xlim(1e8, 1e16)
        plt.xlabel("Energy, eV")

        plt.ylim(1e-13, 1e-10)
        plt.ylabel(r"Flux, $\mathrm{erg~cm^{-2}~s^{-1}}$")

        plt.legend(loc=2, ncol=3)

        plt.tight_layout()
        save_figure("background_contribution")
        plt.show()
    return


if __name__ == '__main__':
    filenames = ["IC_CMB.pck", "IC_Dust.pck", "IC_Starlight.pck", "IC_Total.pck"]
    labels = ["CMB", "Dust", "Starlight", "Total"]
    colors = ["orangered", "seagreen", "royalblue", "black"]
    linestyles = ["dashdot", "solid", "dashed", "solid"]
    # test_tabulated_matrix("IC_CMB.pck")
    # test_interpolated_matrix(filenames[2])
    # test_tabulated_spectrum()
    set_plotting_defaults()
    plt.figure(figsize=(7, 5))
    for i, f in enumerate(filenames):
        ifs = (i == 3)
        test_interpolated_spectrum(f, color=colors[i], linestyle=linestyles[i], label=labels[i], if_show=ifs)
