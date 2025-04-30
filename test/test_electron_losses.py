import numpy as np
from matplotlib import pyplot as plt

import astropy.units as u

from config.units import Gauss
from src.electron_cooling import InverseComptonCooling, SynchrotronCooling, JointCooling


def test_electron_losses():
    energies = np.logspace(9, 19, 1000) * u.eV

    synch = SynchrotronCooling(1e-6 * Gauss)
    icc = InverseComptonCooling()
    joint = JointCooling(1e-6 * Gauss)

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.loglog(energies, icc.power(0, energies.value), color='red', label=f"Inverse Compton losses")
    plt.loglog(energies, synch.power(0, energies.value), color='blue', label=f"Synchrotron losses")

    plt.loglog(energies, joint.power(0, energies.value), color='black', label=f"total")

    plt.xlabel('Energy, eV')
    plt.ylabel('Total losses, eV / yr')
    plt.legend()

    plt.subplot(122)

    plt.plot(energies, -joint.power_derivative(0, energies.value), color='black', label="Total derivative")
    plt.plot(energies, -synch.power_derivative(0, energies.value), color='blue', label='Synchrotron analytical derivative', linestyle='dashed')
    plt.plot(energies, -icc.power_derivative(0, energies.value), color='red', label='Inverse Compton analytical derivative', linestyle='dashed')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
    test_electron_losses()
