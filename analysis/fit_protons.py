import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from astropy.constants import codata2010 as cst
import astropy.units as u
from aafragpy import get_cross_section, get_spectrum

from analysis.electron_synchrotron_only import SynchrotronOnly
from config.plotting import Tab10, save_figure
from config.settings import SPECTRUM_DIR


def get_flux(energy, dn_de, distance, delta_a, delta_delta):
    scaling_factor = 4 * np.pi / (energy * distance * delta_a ** 2 * delta_delta)
    flux = dn_de / scaling_factor
    return flux


def main():
    # set spectrum parameters
    m_p = (cst.m_p * cst.c ** 2).to(u.eV)
    Gamma = 1.8  # power law index
    p_0 = 5e15 * u.eV

    E_proton = np.logspace(12, 19, 500) * u.eV
    p_p = np.sqrt(E_proton ** 2 - m_p ** 2)  # [eV]

    dN_dEp = (p_p / (u.eV)) ** (-Gamma) * np.exp(-p_p / p_0) * (E_proton / p_p)

    data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
    names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data

    synch_only = SynchrotronOnly()

    for i, name in enumerate(names):
        plt.errorbar(e[i], f_cor[i], xerr=[e_l[i], e_p[i]], yerr=[f_l_cor[i], f_p_cor[i]],
                     fmt='o', linestyle='None', uplims=f_p_cor[i] <= 0,
                     color=Tab10[i], label=names[i])

    e_gamma2 = np.logspace(9, 16, 100) * u.eV

    primary_energies_GeV = E_proton.to(u.GeV).value
    secondary_energies_GeV = synch_only.photon_energy.to(u.GeV).value
    secondary_energies_GeV2 = e_gamma2.to(u.GeV).value

    cs_matrix = get_cross_section(secondary='gam', primary_target='p-p',
                                  E_primaries=primary_energies_GeV, E_secondaries=secondary_energies_GeV)

    cs_matrix2 = get_cross_section(secondary='gam', primary_target='p-p',
                                   E_primaries=primary_energies_GeV, E_secondaries=secondary_energies_GeV2)

    spec = get_spectrum(energy_primary=primary_energies_GeV, energy_secondary=secondary_energies_GeV,
                        cs_matrix=cs_matrix[0], prim_spectrum=dN_dEp) * u.mbarn / (u.eV * u.cm ** 3)

    spec2 = get_spectrum(energy_primary=primary_energies_GeV, energy_secondary=secondary_energies_GeV2,
                         cs_matrix=cs_matrix2[0], prim_spectrum=dN_dEp) * u.mbarn / (u.eV * u.cm ** 3)

    # density
    density = 1 / u.cm ** 3  # get_ferriere_density() ????
    dn_dE_dt = spec * density * cst.c
    dn_dE_dt2 = spec2 * density * cst.c

    # SOURCE
    dist = 6.1 * u.kpc
    da = 0.4 * u.deg.to(u.rad)
    dd = 1.0 * u.deg.to(u.rad)
    flux = get_flux(synch_only.photon_energy, dn_dE_dt, distance=dist, delta_a=da, delta_delta=dd)
    flux2 = get_flux(e_gamma2, dn_dE_dt2, distance=dist, delta_a=da, delta_delta=dd)
    e_pred_flux = flux * synch_only.photon_energy

    normalization = 0.8 * np.trapezoid(synch_only.flux, synch_only.photon_energy) / np.trapezoid(e_pred_flux, synch_only.photon_energy)
    avg_val = normalization.to(1)

    print((np.trapezoid(dN_dEp / (u.cm ** 3 * u.eV), E_proton) * avg_val).to(u.cm ** (-3)))
    print((np.trapezoid(E_proton * dN_dEp / (u.cm ** 3 * u.eV), E_proton) * avg_val).to(u.eV * u.cm ** (-3)))

    # plt.loglog(e_phot_values, flux * e_phot_values * normalization, color='k', linestyle='--', label='Proton total')
    plt.loglog(e_gamma2, flux2 * e_gamma2 * normalization, color='k', linestyle='--', label='Proton total')

    plt.yscale('log')
    plt.xscale('log')

    plt.grid(linestyle='--', color='lightgrey')
    plt.legend(loc=2, ncol=1)

    plt.xscale('log')
    plt.xlim(1e9, 1e16)
    plt.xlabel("Energy, eV")

    plt.yscale('log')
    plt.ylim(1e-13, 1e-10)
    plt.ylabel(r"Flux, $\mathrm{erg~cm^{-2}~s^{-1}}$")

    plt.tight_layout()

    save_figure("spectrum_with_a_proton_fit")
    plt.show()

    return


if __name__ == '__main__':
    main()
