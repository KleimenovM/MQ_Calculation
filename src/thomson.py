import numpy as np

from src.black_body_radiation import bbr_density


def thomson_regime_profile_function(x1):
    result = 2 * x1 * np.log(x1) + x1 + 1 - 2 * x1 ** 2
    return result * np.heaviside(1 - x1, 0.5)


def get_thomson_scattered_photons(g1, e1, e2=None):
    """
    g1: gamma of the incident electron
    e1: outgoing particle energies
    e2: incoming particle energies (to integrate)
    """
    if e2 is None:
        e2 = 10 ** np.linspace(-9, -1, 10 ** 3) * u.eV
    e12, e21 = np.meshgrid(e1, e2, indexing='ij')
    x1 = e12 / (4 * g1 ** 2 * e21)
    f1 = thomson_regime_profile_function(x1)
    n_CMB = bbr_density(e21, t_cmb)
    result = 3 * const.sigma_T * const.c / (4 * g1 ** 2) * trapezoid(n_CMB * f1, np.log10(e21.value), axis=1)
    norm = trapezoid(result, e1, axis=0)
    return result / norm