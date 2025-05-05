import numpy as np
from astropy.constants import codata2010 as cst
import astropy.units as u


def bbr_density(energy, temperature):
    """
    Get the BBR density given its temperature
    :param energy: [eV], numpy array / float, energy range
    :param temperature: [K], temperature
    :return: dN/dE
    """
    theta = (energy / (temperature * cst.k_B).to(u.eV)).to('')
    tt = np.array(np.exp(theta), dtype=float)
    np.clip(tt, 1e-32, 1e64)  # cut low tt values
    spec = 1 / (tt - 1)  # spectrum calculation
    if hasattr(energy, "__len__"):
        spec[spec < 1e-20] = 0  # cut low values
    const_factor = 8 * np.pi / (cst.h.to(u.eV * u.s) * cst.c.to(u.cm / u.s)) ** 3
    return const_factor * spec * energy ** 2
