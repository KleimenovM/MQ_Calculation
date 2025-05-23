import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.integrate import quad

import astropy.units as u
from astropy.constants import codata2018 as cst

from config.settings import ISRF_DIR
from config.units import Franklin, Gauss
from src.ebl_photon_density import CMBOnly
from src.klein_nishina import klein_nishina_on_a_given_photon_density_profile
from src.local_density import ISRF


def inverse_compton_timescale(energy, mass):
    # background density
    cmb = CMBOnly()
    isrf = ISRF()
    e_bg = np.logspace(-5, 2, 400) * u.eV  # background photon energy
    d_cmb = cmb.density_e(e_bg, z=0)
    d_isrf = isrf.density_e(e_bg)
    d_bg = d_cmb + d_isrf

    lg_e1_min, lg_e1_max = 5, 19
    lg_e1 = np.linspace(lg_e1_min, lg_e1_max, 10000)
    e1 = 10**lg_e1 * u.eV

    e12, e21 = np.meshgrid(e1, e_bg, indexing='ij')

    ans = np.ones_like(energy.value) * u.eV / u.s
    for i, e_i in enumerate(energy):
        # print(i)
        g_i = (e_i / (mass * cst.c ** 2)).to('')
        ans_i = klein_nishina_on_a_given_photon_density_profile(g1=g_i, e1=e1, e2=e_bg, bg_phot_density=d_bg,
                                                                e12=e12, e21=e21, if_norm=False).to(1 / (u.eV * u.s))
        f = interp1d(lg_e1, e1 * e1 * ans_i, kind='cubic')
        ans[i] = quad(f, lg_e1_min, lg_e1_max)[0] * u.eV / u.s
    return (energy / ans).to(u.year)


def synchrotron_timescale(energy, bfield, mass):
    sin_avg = 2 / 3
    gamma = (energy / (mass * cst.c ** 2)).to('')
    e = cst.e.gauss.value * Franklin
    P_syn = 2 * e ** 4 / (3 * mass ** 2 * cst.c ** 3) * bfield ** 2 * gamma**2 * sin_avg
    return (energy / P_syn).to(u.year)


def diffusion_timescale(energy, length, bfield):
    D = 1e30 * u.cm**2 / u.s * (energy / u.PeV / bfield * 1e-6 * Gauss)**(1/3)
    return (length ** 2 / (2 * D)).to(u.year)


def diffusion_timescale_exp(pwl=1/3):
    energies = np.array([4.5, 45, 141, 224]) * u.TeV
    sizes = np.array([100, 130, 170, 220]) * u.pc
    D = 1e30 * u.cm ** 2 / u.s * (energies / u.PeV)**pwl
    return energies, (sizes ** 2 / D).to(u.year)


if __name__ == '__main__':
    print('Not for direct use')
