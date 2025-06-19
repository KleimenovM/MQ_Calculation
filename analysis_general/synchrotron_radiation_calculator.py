import os
import pickle

import numpy as np
import astropy.units as u

from scipy.integrate import trapezoid

from analysis_general.general_spectrum_analysis import flux_from_particle_emission
from config.settings import INVERSE_COMPTON_DIR, SYNCH_INTERP_DIR
from config.units import uGauss


class SynchrotronRadiationCalculator:
    """
    A class for calculating synchrotron emission using precomputed tables
    """
    def __init__(self, bfield, electron_energy=None, photon_energy=None):
        self.bfield = bfield.to(uGauss)
        path = os.path.join(SYNCH_INTERP_DIR, f"synch_power_avg_interp_{bfield.value * 1e6:.1f}.pck")
        synch_interp_data = pickle.load(open(path, "rb"))

        # electron energies
        self.lg_electron_energy = synch_interp_data[0]
        self.electron_energy = 10 ** self.lg_electron_energy * u.eV

        # photon energies
        self.lg_photon_energy = synch_interp_data[1]
        self.photon_energy = 10 ** self.lg_photon_energy * u.eV

        # synchrotron interpolator
        self.synchrotron_interpolator = synch_interp_data[2]

        # define power matrix
        self.power_matrix = 0. * (1 / (u.eV * u.s))
        # set power matrix
        self.set_power_matrix(electron_energy, photon_energy)

    def set_power_matrix(self, electron_energy=None, photon_energy=None):
        """
        Set (or reset) synchrotron power matrix
        :param electron_energy: [eV], incident electron energy
        :param photon_energy: [eV], emitted photon energy
        :return:
        """
        # change electron energy
        if electron_energy is not None:
            self.lg_electron_energy = np.log10(electron_energy.to(u.eV).value)
            self.electron_energy = electron_energy

        # change photon energy
        if photon_energy is not None:
            self.lg_photon_energy = np.log10(photon_energy.to(u.eV).value)
            self.photon_energy = photon_energy

        # calculate radiation matrix
        lg_xy, lg_yx = np.meshgrid(self.lg_electron_energy,
                                   self.lg_photon_energy,
                                   indexing='ij')
        self.power_matrix = 10 ** self.synchrotron_interpolator((lg_xy, lg_yx)) * (1 / u.s)
        pass

    def photon_luminocity(self, dN_dE):
        """
        Calculate synchrotron photon luminocity E dN/dE.dt
        :param dN_dE: [eV-1], electron spectrum
        :return: [s-1], photon luminocity E dN/dE.dt
        """
        # number of photons homogenously radiated from the nebula (dN/dE.dt)
        photon_lum = trapezoid(self.power_matrix.T * dN_dE, self.electron_energy, axis=1)
        return photon_lum.to(1/u.s)

    def photon_flux(self, dN_dE, dist):
        """
        Calculate synchrotron photon flux emitted from an electron distribution
        :param dN_dE: [eV-1], electron spectrum
        :param dist: [pc], distance to the source
        :return: [erg cm-2 s-1], photon flux
        """
        photon_spectrum = self.photon_luminocity(dN_dE) / self.photon_energy
        flux = flux_from_particle_emission(photon_spectrum=photon_spectrum,
                                           photon_energy=self.photon_energy,
                                           distance=dist)
        return flux


if __name__ == '__main__':
    print("Not for direct use.")
