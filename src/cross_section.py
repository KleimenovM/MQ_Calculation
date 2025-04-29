import numpy as np
from scipy.special import spence

import astropy.units as u
from astropy.constants import codata2010 as const


def beta(e0, e, z, mu):
    """
    Get dimensionless [DL] energy parameter 'beta' from energies e0 (gamma0), e1 (gamma1), redshift z and angle mu
    :param e0: [eV], first incident photon energy
    :param e: [eV], second incident photon energy
    :param z: [DL], redshift
    :param mu: [DL], cos(theta)
    :return: [DL], reaction parameter 'beta'
    """
    m_e2 = (const.m_e * const.c**2)**2
    b = 1.0 - 2.0 * m_e2 / (e0 * e * (1 + z) * (1 - mu))
    return np.sqrt((b + np.abs(b)) / 2.0)


def gamma_gamma_cross_section(e0, e, z, mu):
    """
    Get the gamma + gamma -> e+ e- cross-section (rest frame)
    :param e0: [eV], first incident photon energy
    :param e: [eV], second incident photon energy
    :param z: [DL], redshift
    :param mu: [DL], cos(theta)
    :return: [m2], cross-section
    """
    m_e2 = (const.m_e * const.c ** 2)**2
    b2 = 1.0 - 2.0 * m_e2 / (e0 * e * (1 + z) * (1 - mu))
    b = ((b2 + np.abs(b2)) / 2.0) ** 0.5  # = b2 if b2 > 0; = 0 if b2 < 0
    multiplicator = -4 * b + 2 * b ** 3 + (3 - b ** 4) * np.log((1 + b) / (1 - b))
    return 3 / 16 * const.sigma_T * (1 - b ** 2) * multiplicator


def p(x):
    from scipy.special import spence
    ln2 = np.log(2)
    ln_1mx = np.log(1-x)
    ln_1px = np.log(1+x)
    li2 = spence(1 - (1 - x) / 2)  # see definition of spence function in scipy
    return (ln2**2 - np.pi**2 / 6 + 2 * li2 - x * (1+x**2) / (1 - x**2) + (ln_1px - 2 * ln2) * ln_1mx +
            1/2 * (ln_1mx**2 - ln_1px**2) + (1+x**4) / (2 * (1 - x**2)) * (ln_1px - ln_1mx))


def total_cross_section(e0, z, e):
    """
    Total cross-section integrated by all angles
    :param e0: [eV], first incident photon energy
    :param e: [eV], second incident photon energy
    :param z: [DL], redshift
    :return: [m2], cross-section
    """
    b_max = beta(e0, e, z, -1).value
    m_e2 = (const.m_e * const.c**2)**2
    return 3/4 * const.sigma_T * (1+z)**(-2) * (m_e2 / (e0 * e))**2 * p(b_max) * np.heaviside(b_max, 0.0)


if __name__ == '__main__':
    print("Not for direct use")
