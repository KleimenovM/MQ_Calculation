import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import codata2010 as cst


def single_electron_nucleus_bremsstrahlung(electron_energy, photon_energy, n_nucl, z_nucl = 1.0):
    good_photons = electron_energy > photon_energy
    ph_e = photon_energy[good_photons]

    dim_factor = 3/(2*np.pi) * z_nucl**2 * cst.alpha * cst.sigma_T
    e_i = electron_energy
    e_f = electron_energy - ph_e
    energy_factor = (e_i**2 + e_f**2 - 2/3 * e_i * e_f) / e_i**2
    log_factor = np.log(2 * e_i * e_f / (cst.m_e * cst.c**2 * ph_e)) - 1/2

    result = np.zeros_like(photon_energy.value) * (1 / (u.eV * u.s))
    result[good_photons] = cst.c * n_nucl * dim_factor / ph_e * energy_factor * log_factor

    return result


if __name__ == '__main__':
    print('Not for direct use')
