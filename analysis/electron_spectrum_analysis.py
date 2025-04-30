import os
import pickle
import emcee

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import codata2010 as cst
from scipy.integrate import solve_ivp

from config.units import Gauss
from src.electron_cooling import Cooling
from src.electron_timescales import synchrotron_timescale


def energy_backpropagation(cooling: Cooling, e_timescale=10000 * u.PeV,
                           N_time: int = 10000, N_energy: int = 1000):
    """
    Calculates energy grid backpropagation over time with the provided cooling model.

    The function implements energy loss calculations using a provided
    cooling model and integrates the temporal dynamics of particle
    energies using an ODE solver. It computes results over logarithmic
    time and energy grids defined by the inputs and outputs the grid
    values along with the computed solution.

    :param cooling: An instance of a Cooling class, which defines the cooling mechanisms and power loss function.
    :param e_timescale: The initial energy timescale, with units, used to calculate the time grid.
        The default value is equivalent to 10,000 PeV.
    :param N_time: The number of grid points over the time domain. Default is 10,000.
    :param N_energy: The number of grid points over the energy domain. Default is 1,000.
    :return: A tuple containing the time array, initial energy values, and the computed energy solutions across time.
    """
    magnetic_field = cooling.magnetic_field

    # time grid
    if magnetic_field.value <= 0:
        dt = 10 * u.yr
    else:
        dt = synchrotron_timescale(e_timescale, magnetic_field, cst.m_e)  # unit-time
    print(f"time scale = {dt:.0f}")

    times = np.logspace(0, 5, N_time) * dt
    print(f"time_max = {times[-1]:.0f}")

    # energy grid
    energies = np.logspace(10, 19, N_energy) * u.eV
    print(f"min_energy {energies[0]:.0g}, max_energy {energies[-1]:.0g}")

    # losses
    sol = solve_ivp(cooling.power, t_span=(0, times[-1].value), y0=energies.value,
                    t_eval=times.value, method='DOP853', dense_output=True)

    print(sol.success)

    return sol.t, energies, sol.y



