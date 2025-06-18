import os
import pickle

import numpy as np

from scipy.integrate import trapezoid

import astropy.units as u
from astropy.constants import codata2018 as cst

from config.settings import INVERSE_COMPTON_DIR
from config.units import Franklin, Gauss
from src.background_photon_density import CMBOnly, DustLines, Starlight
from src.klein_nishina import klein_nishina_on_a_given_photon_density_profile
from src.thomson import thomson_on_a_given_photon_density_profile


def inverse_compton_timescale(energy, mass, r=2.1*u.kpc, z=0.5*u.kpc):
    # background density
    density_unit = 1 / (u.eV * u.cm**3)
    e_bg = np.logspace(-6, 2, 800) * u.eV  # background photon energy

    dustlines = DustLines()
    d_dust = dustlines.density_e(e_bg, r, z).to(density_unit)
    stars = Starlight()
    d_stars = stars.density_e(e_bg, r, z).to(density_unit)
    cmb = CMBOnly()
    d_cmb = cmb.density_e(e_bg, .0).to(density_unit)

    d_bg = d_cmb + d_dust + d_stars

    lg_e1_min, lg_e1_max = 5, 19
    lg_e1 = np.linspace(lg_e1_min, lg_e1_max, 4000)
    e1 = 10 ** lg_e1 * u.eV

    e12, e21 = np.meshgrid(e1, e_bg, indexing='ij')

    ans = np.ones_like(energy.value) * u.eV / u.s
    for i, e_i in enumerate(energy):
        g_i = (e_i / (mass * cst.c ** 2)).to('')
        # ans_i = dN(gamma)/dt.de1
        ans_i = klein_nishina_on_a_given_photon_density_profile(g1=g_i, e1=e1, e2=e_bg, bg_phot_density=d_bg,
                                                                e12=e12, e21=e21, if_norm=False).to(1 / (u.eV * u.s))
        ans[i] = np.trapezoid(e1 * ans_i, e1)
    return (energy / ans).to(u.yr)


def inverse_compton_timescale_thomson(energy, mass, r=2.1*u.kpc, z=0.5*u.kpc):
    # background density
    density_unit = 1 / (u.eV * u.cm**3)
    e_bg = np.logspace(-6, 2, 2000) * u.eV  # background photon energy

    dustlines = DustLines()
    d_dust = dustlines.density_e(e_bg, r, z).to(density_unit)
    stars = Starlight()
    d_stars = stars.density_e(e_bg, r, z).to(density_unit)
    cmb = CMBOnly()
    d_cmb = cmb.density_e(e_bg, .0).to(density_unit)

    d_bg = d_cmb + d_dust + d_stars

    lg_e1_min, lg_e1_max = 5, 19
    lg_e1 = np.linspace(lg_e1_min, lg_e1_max, 4000)
    e1 = 10 ** lg_e1 * u.eV

    e12, e21 = np.meshgrid(e1, e_bg, indexing='ij')

    ans = np.ones_like(energy.value) * u.eV / u.s
    for i, e_i in enumerate(energy):
        g_i = (e_i / (mass * cst.c ** 2)).to('')
        # ans_i = dN(gamma)/dt.de1
        ans_i = thomson_on_a_given_photon_density_profile(g1=g_i, e1=e1, e2=e_bg, bg_phot_density=d_bg,
                                                          e12=e12, e21=e21, if_norm=False).to(1 / (u.eV * u.s))
        ans[i] = np.trapezoid(e1 * ans_i, e1)
    return (energy / ans).to(u.yr)


def electron_inverse_compton_timescale(energy, filename="IC_Total.pck"):
    # electron energies
    lg_energy = np.log10(energy.value)

    # photon energies
    lg_e1_min, lg_e1_max = 7, 18
    lg_e1 = np.linspace(lg_e1_min, lg_e1_max, 4000)
    e1 = 10 ** lg_e1 * u.eV

    lg_e12, lg_e21 = np.meshgrid(lg_energy, lg_e1, indexing='ij')

    # ic matrix
    _, _, matrix_interpolator = pickle.load(open(os.path.join(INVERSE_COMPTON_DIR, filename), "rb"))
    values = 10 ** matrix_interpolator((lg_e12, lg_e21)) * (1 / (u.eV * u.s))

    ans = trapezoid(e1 * values, e1, axis=1)
    return (energy / ans).to(u.year)


def synchrotron_timescale(energy, bfield, mass=cst.m_e):
    sin_avg = 2 / 3
    gamma = (energy / (mass * cst.c ** 2)).to('')
    e = cst.e.gauss.value * Franklin
    P_syn = 2 * e ** 4 / (3 * mass ** 2 * cst.c ** 3) * bfield ** 2 * gamma ** 2 * sin_avg
    return (energy / P_syn).to(u.year)


def diffusion_timescale(energy, length, bfield):
    D = 1e30 * u.cm ** 2 / u.s * (energy / u.PeV / bfield * 1e-6 * Gauss) ** (1 / 3)
    return (length ** 2 / (2 * D)).to(u.year)


def diffusion_timescale_exp(pwl=1 / 3):
    energies = np.array([4.5, 45, 141, 224]) * u.TeV
    sizes = np.array([100, 130, 170, 220]) * u.pc
    D = 1e30 * u.cm ** 2 / u.s * (energies / u.PeV) ** pwl
    return energies, (sizes ** 2 / D).to(u.year)


if __name__ == '__main__':
    print('Not for direct use')
