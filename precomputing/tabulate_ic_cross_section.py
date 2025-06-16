import os.path
import pickle

import numpy as np
from astropy import units as u
from astropy.constants import codata2010 as cst
from scipy.interpolate import RegularGridInterpolator

from config.settings import INVERSE_COMPTON_DIR
from src.background_photon_density import CMBOnly, DustLines, Starlight
from src.klein_nishina import klein_nishina_on_a_given_photon_density_profile


def get_background_density(e_bg,
                           r=2.1 * u.kpc,
                           z=0.5 * u.kpc):
    """
    Calculate the background photon density for a given energy range
    :param e_bg: [eV], background photon energy range
    :param r: [kpc], galactocentric radius
    :param z: [kpc], galactocentric height
    :return: [eV-1 cm-3], background photon density (CMB, Dust, Starlight)
    """
    # unit definitions
    e_unit = u.eV
    density_unit = 1 / (u.eV * u.cm ** 3)

    e_bg = e_bg.to(e_unit)

    # CMB background density
    cmb = CMBOnly()
    d_CMB = cmb.density_e(e_bg, z=0).to(density_unit)

    # Dust emission
    dustlines = DustLines()
    d_dust = dustlines.density_e(e_bg, r, z).to(density_unit)

    # Starlight
    starlight = Starlight()
    d_starlight = starlight.density_e(e_bg, r, z).to(density_unit)

    return d_CMB, d_dust, d_starlight


def tabulate_the_spectrum(bg_density, bg_energy, filename,
                          N_e: int = 2000, N_phot: int = 2000):
    """
    Tabulates Inverse Compton photon rate on a homogenous photon background
    and saves it to INVERSE_COMPTON_DIR/filename
    with the use of [Blumenthal, Gould, 1970].
    :param bg_density: [eV-1 cm-3], background photon density
    :param bg_energy: [eV], background photon energy
    :param filename: (str), name of the output file
    :param N_e: (int, default = 2000) electron energy grid size
    :param N_phot: (int, default = 2000) photon energy grid size
    """

    # electrons
    electron_energy = np.logspace(9, 19, N_e) * u.eV
    electron_mass = (cst.m_e * cst.c ** 2).to(u.eV)
    electron_gamma = electron_energy / electron_mass

    # gamma-ray photons
    photon_energy = np.logspace(7, 18, N_phot) * u.eV

    e12, e21 = np.meshgrid(photon_energy, bg_energy, indexing='ij')

    result = np.zeros([N_e, N_phot]) / (u.eV * u.s)

    for i, g1 in enumerate(electron_gamma):
        if i % 50 == 0:
            print(i, end=' ')
        result[i, :] = klein_nishina_on_a_given_photon_density_profile(g1, e1=photon_energy, e2=bg_energy,
                                                                       bg_phot_density=bg_density,
                                                                       e12=e12, e21=e21)
    pickle.dump([electron_energy, photon_energy, result],
                open(os.path.join(INVERSE_COMPTON_DIR, filename), "wb"))

    return


def load_tabulated_matrix(filename):
    """
    Loads the tabulated matrix from INVERSE_COMPTON_DIR/filename
    :param filename: (str), name of the tabulated file
    """
    data = pickle.load(open(os.path.join(INVERSE_COMPTON_DIR, filename), 'rb'))
    return data[0], data[1], data[2]


def interpolate_tabulated_matrix(filename_in, filename_out):
    """
    Loads a table from INVERSE_COMPTON_DIR/filename and creates a RegularGridInterpolator. Saves it to the same folder
    :param filename_in: (str), name of the tabulated file
    :param filename_out: (str), name of the output file
    """
    electron_energy, photon_energy, result = load_tabulated_matrix(filename_in)
    lg_electron_energy = np.log10(electron_energy.to(u.eV).value)
    lg_photon_energy = np.log10(photon_energy.to(u.eV).value)
    lg_matrix = np.log10(result.to(1 / (u.eV * u.s)).value + np.finfo(float).tiny)

    matrix_interp = RegularGridInterpolator((lg_electron_energy, lg_photon_energy), lg_matrix)
    pickle.dump([lg_electron_energy, lg_photon_energy, matrix_interp],
                open(os.path.join(INVERSE_COMPTON_DIR, filename_out), "wb"))
    return


def tabulate_ic_cross_section():
    """
    Tabulate Inverse Compton (IC) cross-section for high-energy (HE) electrons
    on CMB, Infrared (IR) dust photons, Starlight photons and their composition.

    Calculates IC interaction rates with the use of analytical formulae from [Blumental, Gould, 1970].
    Saves them as data tables (lg_E_electron, lg_E_HE_photon, lg_production_rate)

    After tabulation, creates and writes corresponding interpolators to INVERSE_COMPTON_DIR
    """

    # background photons
    N_bg = 2000
    background_energy = np.logspace(-5, 1, N_bg) * u.eV
    d_CMB, d_dust, d_starlight = get_background_density(background_energy)
    d_total = d_CMB + d_dust + d_starlight

    # main calculation cycle
    densities = [d_CMB, d_dust, d_starlight, d_total]
    filenames = ["IC_CMB_t.pck", "IC_Dust_t.pck", "IC_Starlight_t.pck", "IC_Total_t.pck"]

    for i in range(len(filenames)):
        print(f"Tabulating the {filenames[i]}")
        tabulate_the_spectrum(bg_density=densities[i],
                              bg_energy=background_energy,
                              filename=filenames[i])
    return


def interpolate_IC_cross_section():
    # main calculation cycle
    filenames_in = ["IC_CMB_t.pck", "IC_Dust_t.pck", "IC_Starlight_t.pck", "IC_Total_t.pck"]
    filenames_out = ["IC_CMB.pck", "IC_Dust.pck", "IC_Starlight.pck", "IC_Total.pck"]
    for i in range(len(filenames_in)):
        print(f"Interpolating the {filenames_in[i]}")
        interpolate_tabulated_matrix(filename=filenames[i])
    return


if __name__ == '__main__':
    tabulate_ic_cross_section()
