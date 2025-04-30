import os.path
import pickle

import numpy as np
from astropy import units as u
from astropy.constants import codata2010 as cst
from scipy.interpolate import interp1d

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

    return


def load_tabulated_matrix():
    data = pickle.load(open(os.path.join(ELECTRONS_DIR, "spectrum_tabulated.pck"), 'rb'))
    return data[0], data[1], data[2]


if __name__ == '__main__':
    tabulate_the_spectrum()
