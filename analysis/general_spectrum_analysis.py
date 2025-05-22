import numpy as np
import astropy.units as u

from config.units import flux_unit


def volume_approximation(energy):
    # volume (define by Kolmogorov diffusion)
    width_length_ratio = 0.2
    size_1TeV = 100 * u.pc
    size = size_1TeV * (energy / u.TeV) ** (1 / 6)
    return size ** 3 * width_length_ratio ** 2


def prepare_measurements(e, f_cor, f_l_cor, f_p_cor):
    energy = []
    flux, flux_l, flux_p = [], [], []

    for i in range(len(e)):
        for j in range(len(e[i])):
            energy.append(e[i][j].value)
            flux.append(f_cor[i][j].value)
            flux_l.append(f_l_cor[i][j].value)
            flux_p.append(f_p_cor[i][j].value)

    e = np.array(energy) * u.eV
    f, f_l, f_p = np.array(flux) * flux_unit, np.array(flux_l) * flux_unit, np.array(flux_p) * flux_unit

    indices = np.argsort(e)
    return e[indices], f[indices], f_l[indices], f_p[indices]
