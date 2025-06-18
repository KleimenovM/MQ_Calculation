import numpy as np

from scipy.integrate import trapezoid

import astropy.units as u
from astropy.constants import codata2010 as cst

from config.constants import T_CMB, CST_m_e
from src.black_body_radiation import bbr_density


def thomson_regime_profile_function(x1):
    """
    Thomson regime profile [Blumenthal et al., 1970] (2.45)
    :param x1: dimensionless energy x1 = e1/(4e gamma^2)
    :return:
    """
    result = 2 * x1 * np.log(x1) + x1 + 1 - 2 * x1 ** 2
    return result * np.heaviside(1 - x1, 0.5)


def thomson_on_a_given_photon_density_profile(g1, e1, e2, bg_phot_density,
                                              e12=None, e21=None,
                                              if_norm: bool = False, mass=None):
    if e12 is None and e21 is None:
        e12, e21 = np.meshgrid(e1, e2, indexing='ij')

    if mass is None:
        mass = CST_m_e

    x_12 = e12 / (4 * g1**2 * e21)  # x = e1/e1(max)

    f1 = thomson_regime_profile_function(x_12)

    result = 3 * cst.sigma_T * cst.c / (4 * g1**2) * trapezoid(bg_phot_density * f1, np.log(e21 / u.eV), axis=1)

    if if_norm:
        norm = trapezoid(result, e1, axis=0)
        return result / norm

    return result.to(1/(u.eV * u.s))


def thomson_on_CMB(g1, e1, e2=None, if_norm: bool = False):
    """
    Get single Thomson CMB scattering photon density
    g1: gamma of the incident electron
    e1: outgoing photon energies [eV]
    e2: incoming photon energies [eV] (to integrate)
    """

    if e2 is None:
        e2 = 10 ** np.linspace(-9, -1, 10 ** 3) * u.eV

    e12, e21 = np.meshgrid(e1, e2, indexing='ij')

    n_CMB = bbr_density(e21, T_CMB)

    return thomson_on_a_given_photon_density_profile(g1, e1, e2, n_CMB, if_norm=if_norm)


if __name__ == '__main__':
    print("Not for direct use.")
