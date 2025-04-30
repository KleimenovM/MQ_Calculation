import os
import pickle
import numpy as np

import astropy.units as u
from astropy.constants import codata2010 as cst
from scipy.differentiate import derivative
from scipy.interpolate import make_splrep

from config.settings import ELECTRONS_DIR
from config.units import Gauss, Franklin


class Cooling:
    """
    Represents a system or process related to cooling.

    This class provides mechanisms to calculate power and its derivative
    for a cooling process over a given time and energy input. It serves as 
    a base or generic model for systems related to cooling calculations.

    """
    def __init__(self):
        pass

    def power(self, time, energy):
        return

    def power_derivative(self, time, energy):
        return


class SynchrotronCooling(Cooling):
    """
    This class initializes and calculates the total synchrotron radiation power in
    a given magnetic field.

    :param magnetic_field: [uGauss], Strength of the magnetic field
    """
    def __init__(self, magnetic_field):
        super().__init__()
        sin2_avg = 2 / 3
        e = cst.e.gauss.value * Franklin
        sharp_units = u.eV**(-1) * u.yr**(-1)
        self.sharp = (2 * e**4 / (3 * cst.m_e**4 * cst.c**7) * magnetic_field**2 * sin2_avg).to(sharp_units).value

    def power(self, time, energy):
        """
        Synchrotron radiation power.

        :param time: [yr] Time value (no dependece).
        :param energy: [eV] Energy value.
        :return: Synchrotron radiation power [eV/yr]
        """
        return self.sharp * energy**2

    def power_derivative(self, time, energy):
        """
        Derivative of synchrotron radiation power function with respect to energy.

        :param time: [yr] Time value (no dependece).
        :param energy: [eV] Energy value.
        :return: Synchrotron radiation power derivative [1/yr]
        """
        return -2 * self.sharp * energy


class InverseComptonCooling(Cooling):
    """
    This class initializes and calculates the total inverse compton radiation power
    """
    def __init__(self):
        super().__init__()
        self.min_energy = 1e9 * u.eV
        self.max_energy = 1e19 * u.eV
        self.ic_time_f = self.__set_ic_time_approximation()

    @staticmethod
    def __set_ic_time_approximation():
        """
        Load and process inverse Compton (IC) time data and create a spline representation for IC time 
        approximation. This method reads precomputed data from a file, processes the energy and IC 
        time arrays, and generates a spline for efficient interpolation.

        :return: A cubic spline for IC time as a function of log-scaled energy
        """
        data = pickle.load(open(os.path.join(ELECTRONS_DIR, "ic_time.pck"), "rb"))
        energies, ic_time = data[0], data[1]
        ic_time_f = make_splrep(np.log10(energies / u.eV), np.log10(ic_time / u.yr), k=3, s=2)
        return ic_time_f

    def __clip_energy(self, energy_value):
        """
        Clips the given energy value to ensure it stays within the predefined minimum
        and maximum energy bounds.

        :param energy_value: [eV], The energy value to be clipped.
        :return: [eV], The energy value
        """
        return np.clip(energy_value, a_min=self.min_energy.value, a_max=self.max_energy.value)

    def power(self, time, energy):
        """
        Inverse Compton radiation power.

        :param time: [yr] Time value (no dependece).
        :param energy: [eV] Energy value.
        :return: inverse comptopn radiation power [eV/yr]
        """
        energy_value = self.__clip_energy(energy)
        return energy_value / (10**self.ic_time_f(np.log10(energy_value)))
    
    def __loglog_power(self, lg_energy_value):
        """
        Computes the logarithmic value of a power function using base-10 logarithms.
        
        :param lg_energy_value: The logarithmic energy value.
        :return: The base-10 logarithm of the computed power.
        """
        return np.log10(self.power(0.0, 10**lg_energy_value))

    def power_derivative(self, time, energy):
        """
        Derivative of Inverse Compton radiation power function with respect to energy.
        This function calculates double-log derivative of the power function.

        :param time: [yr] Time value (no dependece).
        :param energy: [eV] Energy value.
        :return: Inverse Compton radiation power derivative [1/yr]
        """
        energy_value = self.__clip_energy(energy)
        lg_energy_value = np.log10(energy_value)
        power = self.power(time, energy_value)
        return -power / energy_value * derivative(self.__loglog_power, lg_energy_value).df


class JointCooling(Cooling):
    def __init__(self, magnetic_field=1e-6 * Gauss):
        super().__init__()
        self.synch = SynchrotronCooling(magnetic_field)
        self.ic = InverseComptonCooling()

    def power(self, time, energy):
        return self.synch.power(time, energy) + self.ic.power(time, energy)

    def power_derivative(self, time, energy):
        return self.synch.power_derivative(time, energy) + self.ic.power_derivative(time, energy)


if __name__ == '__main__':
    print("Not for direct use")
