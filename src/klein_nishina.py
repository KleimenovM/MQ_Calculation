import numpy as np
from astropy.constants import codata2010 as const
import astropy.units as u
from scipy.integrate import trapezoid

from config.constants import T_CMB
from src.black_body_radiation import bbr_density


def klein_nishina_profile_function_x(x1, g_e1):
    q1 = x1 / (1 + g_e1 - g_e1 * x1)
    return klein_nishina_profile_function(q1, g_e1) * (abs(x1 - 0.5) < 0.5)


def klein_nishina_profile_function(q1, g_e1):
    q1 = q1.clip(1e-32, 1)
    result = (2 * q1 * np.log(q1) + (1 + 2 * q1) * (1 - q1) + 1 / 2 * (g_e1 * q1) ** 2 / (1 + g_e1 * q1) * (1 - q1))
    return result


def klein_nishina_on_a_given_photon_density_profile(g1, e1, e2, bg_phot_density,
                                                    e12=None, e21=None,
                                                    if_norm: bool = False, mass=None):
    """
    Calculate the Klein-Nishina scattering on a given photon density profile
    :param g1: [DL], electron gamma

    :param e1: [eV], outgoing photon energy
    :param e2: [eV], incoming photon energy
    :param e12: [eV], e1 x e2 matrix
    :param e21: [eV], e2 x e1 matrix

    :param bg_phot_density: [cm-3 eV-1] background photon density profile
    :param if_norm: normalizes the result to the total scattering rate
    :param mass: [g], mass of the particle

    :return: dN / dt de1
    """
    if e12 is None and e21 is None:
        e12, e21 = np.meshgrid(e1, e2, indexing='ij')

    if mass is None:
        mass = (const.m_e * const.c ** 2).to(u.eV)

    g_e21 = 4 * e21 * g1 / mass
    E12 = e12 / (g1 * mass)
    x_12 = E12 * (1 + g_e21) / g_e21

    F1 = klein_nishina_profile_function_x(x_12, g_e21)

    result = 4 * const.sigma_T * const.c / (3 * g1 ** 2) * trapezoid(bg_phot_density * F1, np.log(e21 / u.eV), axis=1)

    if if_norm:
        norm = trapezoid(result, e1, axis=0)
        return result / norm

    return result


def klein_nishina_on_CMB(g1, e1, e2=None, e12=None, e21=None, if_norm: bool = False):
    """
    Get single Klein-Nishina CMB scattering photon density
    g1: gamma of the incident electron
    e1: outgoing photon energies [eV]
    e2: incoming photon energies [eV] (to integrate)
    """

    if e2 is None:
        e2 = 10 ** np.linspace(-9, -1, 10 ** 3) * u.eV

    e12, e21 = np.meshgrid(e1, e2, indexing='ij')

    n_CMB = bbr_density(e21, T_CMB)

    return klein_nishina_on_a_given_photon_density_profile(g1, e1, e2, n_CMB, if_norm=if_norm)
