import os
import pickle

import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from config.settings import ELECTRONS_DIR
from src.synchrotron_emission import single_electron_synchrotron_emission_power
from config.units import Gauss


def show_synch_table(electron_energy, photon_energy, bfield, table):
    plt.pcolormesh(electron_energy.value, photon_energy.value, np.log10(table.value + np.finfo(float).tiny).T,
                   cmap='rainbow', vmin=-10, vmax=-1)
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar()
    plt.xlabel('Electron Energy [eV]')
    plt.ylabel('Photon Energy [eV]')
    plt.show()
    return


def save_synch_table(electron_energy, photon_energy, bfield, table):
    lg_table = np.log10(table.value + np.finfo(float).tiny)
    lg_electron_energy = np.log10(electron_energy.value)
    lg_photon_energy = np.log10(photon_energy.value)

    path = os.path.join(ELECTRONS_DIR, f"synch_power_{bfield.value * 1e6:.1f}.pck")
    pickle.dump([lg_electron_energy, lg_photon_energy, lg_table], open(path, "wb"))
    return


def tabulate_synch_emission(electron_energy, photon_energy, bfield):
    result = np.zeros([electron_energy.size, photon_energy.size]) * (1 / u.s)
    for i, e_i in enumerate(electron_energy):
        result[i, :] = single_electron_synchrotron_emission_power(e_i, photon_energy, bfield)

    # show_table(electron_energy, photon_energy, bfield, result)
    save_synch_table(electron_energy, photon_energy, bfield, result)
    return


def interpolate_tabulated_synchrotron(bfield):
    path_in = os.path.join(ELECTRONS_DIR, f"synch_power_{bfield.value * 1e6:.1f}.pck")
    data = pickle.load(open(path_in, "rb"))

    lg_ee, lg_pe, lg_table = data[0], data[1], data[2]
    lg_interpolator = RegularGridInterpolator((lg_ee, lg_pe), lg_table)

    path_out = os.path.join(ELECTRONS_DIR, f"synch_power_interp_{bfield.value * 1e6:.1f}.pck")
    pickle.dump([lg_ee, lg_pe, lg_interpolator], open(path_out, "wb"))

    return


if __name__ == '__main__':
    bf = 8e-6 * Gauss
    # ee = np.logspace(6, 19, 1000) * u.eV
    # phe = np.logspace(-8, 15, 2000) * u.eV
    # tabulate_synch_emission(ee, phe, bf)
    interpolate_tabulated_synchrotron(bf)

