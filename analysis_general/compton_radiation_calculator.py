import os
import pickle

import numpy as np
import astropy.units as u

from scipy.integrate import trapezoid

from analysis_general.general_spectrum_analysis import flux_from_particle_emission
from config.settings import INVERSE_COMPTON_DIR


class ComptonRadiationCalculator:
    """
    A class for calculating Inverse Compton emission using precomputed tables
    on local stellar density (IC_Total.pck), CMB (IC_CMB.pck), Dust (IC_Dust.pck),
    and Starlight (IC_Starlight.pck).
    """
    def __init__(self, electron_energy=None, photon_energy=None,
                 filename: str = "IC_Total.pck"):
        data = pickle.load(open(os.path.join(INVERSE_COMPTON_DIR, filename), "rb"))

        # electron energies
        self.lg_electron_energy = data[0]
        self.electron_energy = 10 ** self.lg_electron_energy * u.eV

        # photon energies
        self.lg_photon_energy = data[1]
        self.photon_energy = 10 ** self.lg_photon_energy * u.eV

        # compton interpolator
        self.compton_interpolator = data[2]

        # define radiation matrix
        self.radiation_matrix = 0. * (1 / (u.eV * u.s))
        # set radiation matrix
        self.set_radiation_matrix(electron_energy, photon_energy)
        pass

    def set_radiation_matrix(self, electron_energy=None, photon_energy=None):
        """
        Set (or reset) IC radiation matrix
        :param electron_energy: [eV], incident electron energy
        :param photon_energy: [eV], scattered photon energy
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
        self.radiation_matrix = 10 ** self.compton_interpolator((lg_xy, lg_yx)) * (1 / (u.eV * u.s))
        pass

    def photon_emission(self, dN_dE):
        """
        Calculate IC photon number emitted per second per energy dN/dE.dt
        :param dN_dE: [eV-1], electron spectrum
        :return: [eV-1 s-1], photon emission
        """
        # number of photons homogenously radiated from the nebula (dN/dE.dt)
        photon_spectrum = trapezoid(self.radiation_matrix.T * dN_dE, self.electron_energy)
        return photon_spectrum.to(1/(u.eV * u.s))

    def photon_flux(self, dN_dE, dist):
        """
        Calculate IC photon flux emitted from an electron distribution
        :param dN_dE: [eV-1], electron spectrum
        :param dist: [pc], distance to the source
        :return: [erg cm-2 s-1], photon flux
        """
        photon_spectrum = self.photon_emission(dN_dE)
        flux = flux_from_particle_emission(photon_spectrum=photon_spectrum,
                                           photon_energy=self.photon_energy,
                                           distance=dist)
        return flux


if __name__ == '__main__':
    print("Not for direct use.")
