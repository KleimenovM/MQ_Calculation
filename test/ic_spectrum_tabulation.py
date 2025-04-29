import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from astropy import units as u
from astropy.constants import codata2010 as cst

from config.settings import ELECTRONS_DIR
from src.ebl_photon_density import CMBOnly
from src.klein_nishina import klein_nishina_on_a_given_photon_density_profile


def get_background_density(e_bg):
    """
    Calculate the background photon density for a given energy range
    :param e_bg: [eV], background photon energy range
    :return: [eV-1 cm-3], background photon density (CMB + IR)
    """
    # unit definitions
    e_unit = u.eV
    density_unit = u.eV ** (-1) * u.cm ** (-3)

    # infrared background density upload
    lg_e_bg = np.log10(e_bg.to(e_unit).value)
    with open("local_density.pck", 'rb') as f:
        e_bg1, e_d_bg1 = pickle.load(f)
    e_bg1 = np.flip(e_bg1)
    e_d_bg1 = np.flip(e_d_bg1)
    d_bg1 = e_d_bg1 / e_bg1

    # infrared background density interpolation function
    lg_e_bg1 = np.log10(e_bg1.to(e_unit).value)
    lg_d_bg1 = np.log10(d_bg1.to(density_unit).value)
    f_IR = interp1d(lg_e_bg1, lg_d_bg1, kind='linear', fill_value='extrapolate')

    # CMB background density calculation
    cmb = CMBOnly()
    d_bg_CMB = cmb.density_e(e_bg, z=0)
    d_bg_IR = 10 ** f_IR(lg_e_bg) * density_unit

    return d_bg_IR + d_bg_CMB


def tabulate_the_spectrum():
    # electrons
    N_e = 2000
    electron_energy = np.logspace(9, 19, N_e) * u.eV
    electron_mass = (cst.m_e * cst.c ** 2).to(u.eV)
    electron_gamma = electron_energy / electron_mass

    # gamma-ray photons
    N_phot = 2000
    photon_energy = np.logspace(7, 18, N_phot) * u.eV

    # background photons
    N_bg = 2000
    background_energy = np.logspace(-6, 0, N_bg) * u.eV
    background_density = get_background_density(background_energy)

    e12, e21 = np.meshgrid(photon_energy, background_energy, indexing='ij')

    result = np.zeros([N_e, N_phot]) / (u.eV * u.s)

    for i, g1 in enumerate(electron_gamma):
        if i % 50 == 0:
            print(i, end=' ')
        result[i, :] = klein_nishina_on_a_given_photon_density_profile(g1, e1=photon_energy, e2=background_energy,
                                                                       bg_phot_density=background_density,
                                                                       e12=e12, e21=e21)
    pickle.dump([electron_energy, photon_energy, result],
                open(os.path.join(ELECTRONS_DIR, "spectrum_tabulated.pck"), "wb"))

    # test matrix
    """xx, yy = np.meshgrid(electron_energy.value, photon_energy.value, indexing='ij')
    plt.pcolormesh(xx, yy, np.log10(result.value + np.finfo(float).tiny), vmin=-32)
    plt.colorbar()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()"""

    # Test spectrum
    """spec = ElectronSpectrumParametrization(n0=1.0 / (u.eV * u.cm ** 3), e0=1e12 * u.eV,
                                           eta0=0.0, p0=2, k10=0.0, k20=3.0)

    e_phot_values, corrected_fluxes_values, corrected_errors_values = get_energies_and_fluxes("spectrum_UHE.pck",
                                                                                              if_plot=True)

    spec_vals = spec.dn_de0(electron_energy)
    print((spec_vals * result.T).shape)
    photon_spectrum = trapezoid(spec_vals * result.T, electron_energy, axis=1)
    value = max(corrected_fluxes_values) / max(photon_energy**2 * photon_spectrum)
    plt.loglog(photon_energy, 0.2 * photon_energy**2 * photon_spectrum * value)
    plt.xlim(1e8, 1e16)
    plt.ylim(1e-14, 1e-10)
    plt.show()"""

    return


def save_tabulated_spectrum():
    return


if __name__ == '__main__':
    tabulate_the_spectrum()
    # save_tabulated_spectrum()
