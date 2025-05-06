import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import codata2010 as cst

from scipy.special import gamma, kv

from config.constants import CST_HC, CST_e
from config.units import Gauss, Franklin


def first_synchrtoron_function_approximation(x):
    # analytical approximation taken from [Fouka, Ouichaoui, 2013] (relative error < 0.26%)
    a1 = np.array([-0.97947838884478688, -0.83333239129525072, 0.15541796026816246])
    a2 = np.array([-4.69247165562628882e-2, -0.70055018056462881, 1.03876297841949544e-2])

    # theta function approximations: delta_1 = 1 for x << 1, delta_2 = 1 for x >> 1 (0 else)
    # index arrays
    k1 = np.arange(1, 4, 1)
    k2 = np.arange(1, 4, 1)

    x0, _ = np.meshgrid(x, k1, indexing='ij')

    H1 = np.sum(a1 * x0**(1/k1), axis=1)
    H2 = np.sum(a2 * x0**(1/k2), axis=1)

    delta_1 = np.exp(H1)
    delta_2 = 1 - np.exp(H2)

    # the asymptotes
    F1 = np.pi * 2**(5/3) / (np.sqrt(3) * gamma(1/3))
    F2 = np.sqrt(np.pi/2)

    F_low = F1 * x**(1/3)
    F_high = F2 * np.exp(-x) * x**(1/2)

    # return the convolution
    return delta_1 * F_low + delta_2 * F_high


def single_electron_synchrotron_emission_power(electron_energy, photon_energy, bfield):
    e_c = cst.h * cst.c / (4 * np.pi) * 3 * CST_e * bfield * electron_energy**2 / (cst.m_e * cst.c**2)**3
    x = (photon_energy / e_c).to('')  # [DL]
    p_dl = first_synchrtoron_function_approximation(x)
    dim_factor = np.sqrt(3) * CST_e**3 * bfield / (cst.h * cst.m_e * cst.c**2)
    return dim_factor * p_dl


if __name__ == '__main__':
    print('Not for direct use')

