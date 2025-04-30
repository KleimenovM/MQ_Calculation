import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from config.settings import ISRF_DIR, SPECTRUM_DIR
from config.units import flux_unit
from spectrum.gamma_ray_measurements import gamma_ray_data
from src.cross_section import total_cross_section
from src.ebl_photon_density import CMBOnly


def fix_lower(x, data, brd=1e-10):
    """
    Adjusts the input data to exclude values below a specified threshold.

    :param x: Array-like or iterable containing elements to filter.
    :param data: An object with a `value` attribute that supports
        comparison with the threshold (`brd`), such as a NumPy array.
    :param brd: A lower boundary threshold (default is `1e-10`) used to
        determine which values in `data.value` are retained.

    :return: A tuple containing two filtered subsets:
        1. Elements of `x` corresponding to `True` results in the boolean
           condition.
        2. Values from `data` that are greater than `brd`.
    """
    trues = data.value > brd
    return x[trues], data[trues]


def get_extinction_at_fixed_energy(photon_energy, distance,
                                   ir_energies, ir_density,
                                   cmb_energies, cmb_density):
    """
    Calculates the extinction (attenuation factor) for a photon at a fixed energy due
    to scattering and absorption by the infrared (IR) and cosmic microwave background
    (CMB) photons. The extinction is computed using the optical depth (`tau`) for both
    the IR and CMB backgrounds and returns the resulting exponential attenuation factor.

    :param photon_energy: [eV] Photon energy for which the extinction needs to be calculated.
    :param distance: [kpc], Distance over which the photon propagates.
    :param ir_energies: [eV], Energies of the IR photons.
    :param ir_density: [eV], Number densities corresponding to the IR photon energies.
    :param cmb_energies: [eV], Energies of the CMB photons.
    :param cmb_density: [eV], Number densities corresponding to the CMB photon energies.
    :return: [DL] Exponential attenuation factor.
    """

    tcs_1 = total_cross_section(e0=photon_energy, e=ir_energies, z=0).to(u.cm ** 2)
    tcs_2 = total_cross_section(e0=photon_energy, e=cmb_energies, z=0).to(u.cm ** 2)

    tau_ir = np.abs(np.trapezoid(tcs_1 * ir_density * ir_energies * distance, np.log(ir_energies.value)))
    tau_cmb = np.abs(np.trapezoid(tcs_2 * cmb_density * cmb_energies * distance, np.log(cmb_energies.value)))

    return np.exp(tau_ir + tau_cmb)


def take_extinction_into_account(gamma_ray_energies, distance):
    """
    Calculates the extinction of gamma rays over a given distance by taking into account the ISFR density map and the
    CMB density

    :param gamma_ray_energies: [eV], gamma-ray energies for which to compute the extinction.
    :param distance: [kpc], Distance over which the extinction is calculated, expressed in appropriate units.
    :return: [DL], extinction values corresponding to the input gamma-ray energies.
    """
    data = pickle.load(open(os.path.join(ISRF_DIR, "density_map.pck"), "rb"))
    energies = data[0]
    densities = data[1] / distance

    e1, d1 = fix_lower(energies, densities)

    cmb = CMBOnly()
    energies2 = 10**np.linspace(-6, 0, 100) * u.eV
    cmb_density = cmb.density_e(energies2, z=0)
    e2, d2 = fix_lower(energies2, cmb_density)

    n = gamma_ray_energies.shape[0]
    res = np.zeros(n)

    for i, gre in enumerate(gamma_ray_energies):
        res[i] = get_extinction_at_fixed_energy(gre, distance, e1, d1, e2, d2)

    return res


def save_the_spectrum():
    distance = 6.6 * u.kpc  # GAIA distance to V4641

    names, e, f, f_l, f_p, e_l, e_p = gamma_ray_data()
    f_cor, f_l_cor, f_p_cor = [], [], []

    for i, name in enumerate(names):
        print(e[i].unit)
        extinction_factor_i = take_extinction_into_account(e[i], distance)
        f_cor.append(f[i] * extinction_factor_i)
        f_l_cor.append(f_l[i] * extinction_factor_i)
        f_p_cor.append(f_p[i] * extinction_factor_i)

    data = [names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p]
    pickle.dump(data, open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "wb"))
    return


if __name__ == '__main__':
    save_the_spectrum()
