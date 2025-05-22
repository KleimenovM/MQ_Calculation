import os
import pickle
import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from analysis.general_spectrum_analysis import volume_approximation
from config.plotting import set_plotting_defaults
from config.settings import ELECTRONS_DIR
from config.units import Gauss
from src.electron_spectrum_parametrization import SpectrumParametrization
from src.synchrotron_emission import electron_synchrotron_emission_luminosity
from tabulate_synch_emission import show_synch_table


def get_the_table(bfield):
    path = os.path.join(ELECTRONS_DIR, f"synch_power_{bfield.value * 1e6:.1f}.pck")
    data = pickle.load(open(path, "rb"))
    electron_energy, photon_energy, table = 10 ** data[0] * u.eV, 10 ** data[1] * u.eV, 10 ** data[2] * u.s ** (-1)
    # show_synch_table(electron_energy, photon_energy, bfield, table)
    return electron_energy, photon_energy, bfield, table


def main(bfield):
    electron_energy, photon_energy, bfield, table = get_the_table(bfield)

    sp0 = SpectrumParametrization(n0=1e27*u.cm**(-3), e0=1e12 * u.eV,
                                  eta0=0.0, p0=1.8, k10=-10.0, k20=2.9)

    electron_volume = volume_approximation(electron_energy)

    t0 = time.time()
    lum_straight = electron_synchrotron_emission_luminosity(electron_energy=electron_energy,
                                                            electron_density=sp0.dn_de0(electron_energy),
                                                            electron_volume=electron_volume,
                                                            photon_energy=photon_energy,
                                                            bfield=bfield)
    t1 = time.time()

    lum_conv = np.trapezoid(table.T * sp0.dn_de0(electron_energy) * electron_volume, electron_energy, axis=1)
    t2 = time.time()
    print(f"furst in {t1-t0:.3f} seconds, second in {t2-t1:.3f} seconds")

    set_plotting_defaults()
    plt.plot(photon_energy, lum_straight)
    plt.plot(photon_energy, lum_conv)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return


if __name__ == '__main__':
    bf = 2e-6 * Gauss
    main(bf)
