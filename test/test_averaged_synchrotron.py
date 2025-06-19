import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from config.plotting import set_plotting_defaults, Tab20
from config.units import uGauss
from src.synchrotron_emission import (single_electron_synchrotron_emission_power_symmetrized,
                                      single_electron_synchrotron_emission_power)


def test_averaged_synchrotron_emission_power():
    electron_energy = np.array([1e13, 1e14, 1e15]) * u.eV
    magnetic_field = 1 * uGauss

    synch_phot_energy = np.logspace(-2, 12, 1000) * u.eV

    set_plotting_defaults()
    for i, e_i in enumerate(electron_energy):
        synch_flux = (synch_phot_energy * single_electron_synchrotron_emission_power(e_i,
                                                                                     synch_phot_energy,
                                                                                     magnetic_field)).to(u.erg / u.s)

        synch_flux_avg = (synch_phot_energy * single_electron_synchrotron_emission_power_symmetrized(
            e_i, synch_phot_energy, magnetic_field
        )).to(u.erg/u.s)

        plt.loglog(synch_phot_energy, synch_flux, label=f"{e_i.to(u.TeV):.0f}",
                   color=Tab20[2*i+1], linestyle='dashed')
        plt.loglog(synch_phot_energy, synch_flux_avg, color=Tab20[2*i], label=f"{e_i.to(u.TeV):.0f}, avg")

    plt.ylim(1e-14, 1e-6)
    plt.legend(ncols=3)

    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    test_averaged_synchrotron_emission_power()
