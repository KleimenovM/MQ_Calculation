import os
import pickle
import numpy as np

import astropy.units as u
from scipy.interpolate import interp1d

from config.settings import ISRF_DIR


class ISRF:
    """
    Stores density of the Interstellar Radiation (ISRF), which comprises
    dust infrared emission and stellar optical emission (starlight)
    Based on [Misiriotis, 2006] and [Vernetto, 2016]
    """
    def __init__(self):
        # import dust density
        path_dust = os.path.join(ISRF_DIR, 'local_dust_emission_density.pck')
        energy_dust, dust_energy_density = pickle.load(open(path_dust, 'rb'))
        self.lg_dust_energy_density = self.log_interpolate(energy_dust, dust_energy_density)

        # import starlight density
        path_stars = os.path.join(ISRF_DIR, 'local_starlight_density.pck')
        energy_stars, star_energy_density = pickle.load(open(path_stars, 'rb'))
        self.lg_star_energy_density = self.log_interpolate(energy_stars, star_energy_density)

    @staticmethod
    def log_interpolate(x, y):
        """
        Log-interpolate lg(y) on lg(x) without bounds error with extrapolation.
        :param x: x-values (with units)
        :param y: y-values (with units)
        :return: lg-lg scipy interpolator
        """
        lg_x = np.log10(x.value)
        lg_y = np.log10(y.value)

        interpolator = interp1d(lg_x, lg_y, bounds_error=False, fill_value="extrapolate")
        return interpolator

    def density_e(self, energy):
        """
        Calculate background photon density at the given energy.
        :param energy: [eV], background photon energy
        :return: [eV-1 cm-3], background photon density
        """
        lg_e = np.log10(energy.value)
        dust_energy_density = 10**self.lg_dust_energy_density(lg_e) * u.cm**(-3)
        star_energy_density = 10**self.lg_star_energy_density(lg_e) * u.cm**(-3)
        energy_density = dust_energy_density + star_energy_density
        return energy_density / energy


if __name__ == '__main__':
    print("Not for direct use.")
