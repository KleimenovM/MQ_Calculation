import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import codata2010 as const
import astropy.units as u
import matplotlib.colors

from analysis_electrons.spectrum_fit_electrons import JointSteadyStateFit
from analysis_general.synchrotron_radiation_calculator import SynchrotronRadiationCalculator

from config.plotting import set_plotting_defaults
from config.settings import SYNCH_INTERP_DIR
from config.units import uGauss, Gauss, flux_unit
from precomputing.tabulate_ic_cross_section import get_background_density
from spectrum.ic_spectrum_modification import plot_he_spectrum
from spectrum.measurements_xray import plot_x_ray_constraints

sys.path.append("../../libpython/")
import am3


def initialize():
    # 2.1. Initialize AM3 (run only once)
    print("--- initialize ---")
    return am3.AM3()


def set_switches(my_am3: am3.AM3):
    print("--- set_switches IN ---")
    # 2.2. SET SWITCHES
    # Manually define max proton and electron energy
    my_am3.set_estimate_max_energies(0)

    # Turn on for keeping track of the different SED components, for plotting purposes;
    # turn off for efficiency.
    my_am3.set_process_parse_sed(1)

    # Hadronic processes on
    my_am3.set_process_hadronic(0)

    # Keep track of positrons and electrons separately
    my_am3.set_process_merge_positrons_into_electrons(0)

    # Escape (see documentation)
    my_am3.set_process_escape(0)

    # Expansion of radiation zone (see documentation)
    my_am3.set_process_expansion(0)
    my_am3.set_process_adiabatic_cooling(0)

    # Electron synchrotron and synchrotron self-absorption
    my_am3.set_process_electron_syn(1)  # both emission and cooling
    my_am3.set_process_ssa(0)

    # Proton synchrotron - subdominant in this case, but may be relevant for B>~10 G
    my_am3.set_process_proton_syn(0)

    # Quantum synchrotron for pairs off by default.
    # In AGN simulations this effect can typically be neglected, so turn off for efficiency
    my_am3.set_process_quantum_syn(0)

    # Inverse Compton by electrons and protons
    my_am3.set_process_electron_compton(1)  # both emission and cooling
    my_am3.set_process_proton_compton(0)

    # Direct Compton turned off by default
    my_am3.set_process_compton_photon_energy_loss(0)

    # Synchrotron and inverse Compton by muons and pions - off by default.
    # In AGN simulations this effect can typically be neglected, so turn off for efficiency
    my_am3.set_process_muon_syn(0)
    my_am3.set_process_pion_syn(0)
    my_am3.set_process_muon_compton(0)
    my_am3.set_process_pion_compton(0)

    # Secondary particle decay
    my_am3.set_process_pion_decay(0)
    my_am3.set_process_muon_decay(0)

    # Photon annihilation (gamma gamma -> e- e+)
    my_am3.set_process_annihilation(0)
    my_am3.set_optimize_annihilation_pair_emission(0)  # optimization

    # Bethe-Heitler pair production (p gamma -> p e+ e-)
    my_am3.set_process_bethe_heitler(0)
    my_am3.set_optimize_bethe_heitler_outgoing_pairs_grid(0)
    my_am3.set_optimize_bethe_heitler_incoming_protons_min(1e10)
    my_am3.set_optimize_bethe_heitler_target_photon_max(1e6)

    # Photo-pion production (nucleon gamma -> nucleon pion)
    my_am3.set_process_photopion(0)
    my_am3.set_optimize_photopion_target_photon_grid(0)
    my_am3.set_optimize_photopion_target_photon_max(1e6)

    my_am3.set_profile_timing(0)  # Used for profiling, as in the bottom plot. Turn off for efficiency.

    print("--- set_switches OUT ---")

    # ---------------
    # 2.3. INITIALIZE INTEGRATION KERNELS
    # ---------------

    # init kernels
    print("--- init kernels IN ---")
    my_am3.init_kernels()
    print("--- init kernels OUT ---")
    return my_am3


def set_injection_parameters(my_am3: am3.AM3, lifetime=10 * u.kyr,
                             radius=1e3 * u.pc, bfield=1 * uGauss,
                             spectrum=None, params=None):
    print("--- injection parameters IN ---")
    """
    1. SOURCE AND INJECTION PARAMETERS
    """
    volume = 4 / 3. * np.pi * radius ** 3

    # set magnetic field
    my_am3.set_mag_field(bfield.to(Gauss).value)

    # set time step
    time_step = 20 * u.yr
    my_am3.set_solver_time_step(time_step.to(u.s).value)

    if spectrum is None or params is None:
        injpower_e = 5e36 * u.erg / u.s  # [erg/s]

        m_e_eV = 1e9 * u.eV  # minimal energy
        e_max = 1e15 * u.eV  # maximal energy
        gamma_e = 2
        exp_factor = 1.0
        my_am3.set_powerlaw_injection_parameters_electrons(
            volume.to(u.cm ** 3).value,  # volume (cm3)
            injpower_e.to(u.erg / u.s).value,  # injected power [erg/s]
            m_e_eV.to(u.eV).value,  # minimum electron energy [eV]
            m_e_eV.to(u.eV).value,  # break electron energy [eV]
            e_max.to(u.eV).value,  # max electron energy [eV]
            gamma_e,  # pwl index before the break [DL]
            gamma_e,  # pwl index after the break [DL]
            exp_factor)  # exp attenuation factor [DL]
    else:
        el_e = my_am3.get_egrid_lep() * u.eV
        eta, gamma, k10, k20 = params[0], params[1], -3.0, params[2]
        my_am3.set_injection_rate_electrons(
            (el_e * spectrum.dn_de(el_e, eta, gamma, k10, k20) / volume / lifetime).to(1 / (u.cm ** 3 * u.s)).value
        )

    print("- electrons set -")

    print("- photons set -")
    print("--- injection parameters OUT ---")
    return my_am3


def evolve_the_system(my_am3: am3.AM3, lifetime=10 * u.kyr, radius=1e3 * u.pc):
    print("--- evolve the system IN ---")
    dt = my_am3.get_solver_time_step() * u.s
    N = int((lifetime / dt).to('').value)
    print(N)

    # BACKGROUND PHOTONS
    egrid_jetframe = my_am3.get_egrid_photons() * u.eV
    ee = my_am3.get_egrid_lep() * u.eV

    volume = 4 / 3. * np.pi * radius ** 3

    bgd = get_background_density(e_bg=egrid_jetframe)
    photon_density = bgd[0] + bgd[1] + bgd[2]
    photon_e_dn_de = photon_density * egrid_jetframe

    for i in range(N):
        e_d = my_am3.get_electrons()  # collect electrons
        # throw away all the particles generated by the nebula!!!
        my_am3.clear_particle_densities()
        my_am3.set_current_densities_electrons(e_d)

        if (i - 1) % (N // 5) == 0:
            print(i, end=' ')
            # plt.subplot(1, 2, 2)
            # e_d1 = my_am3.get_electrons()
            # plt.loglog(ee, (ee * (e_d1 * 1 / u.cm ** 3) * volume * N / i).to(u.eV))

        my_am3.set_current_densities_photons(photon_e_dn_de.to(1 / u.cm ** 3).value)
        my_am3.evolve_step()

    print("finish")

    return my_am3


def retreive_the_components(my_am3: am3.AM3, lifetime=10 * u.kyr, radius=1e3 * u.pc):
    egrid_jetframe = my_am3.get_egrid_photons() * u.eV
    bgd = get_background_density(e_bg=egrid_jetframe)
    photon_density = bgd[0] + bgd[1] + bgd[2]
    photon_e_dn_de = photon_density * egrid_jetframe

    gammas_components = np.array([my_am3.get_photons()]) * (u.cm ** (-3)) - photon_e_dn_de  # [cm-3]

    # ---------------
    # 5.2 Define conversion to observed quantities
    # ---------------
    grid_gammas = my_am3.get_egrid_photons() * u.eV  # [eV]

    Lobs = np.zeros_like(gammas_components.value) * u.erg / (u.cm ** 2 * u.s)

    last_step = my_am3.get_solver_time_step() * u.s

    dist = 6.1 * u.kpc
    area = 4 * np.pi * dist ** 2
    volume = 4 / 3. * np.pi * radius ** 3

    for i in range(len(gammas_components)):
        Lobs[i] = (grid_gammas * gammas_components[i] * volume / last_step / area).to(flux_unit)

    names = ['primary ele (sy)', 'primary ele (ic)']

    plt.subplot(1, 2, 1)
    # plot spectral components
    for i in range(0, len(gammas_components)):
        plt.plot(grid_gammas, Lobs[i], alpha=1.0, label=names[i])

    plot_he_spectrum(plt.gca())
    plot_x_ray_constraints(plt.gca())

    plt.xlim(1e0, 1e17)
    plt.ylim(1e-14, 1e-10)
    plt.ylabel(r'$\epsilon^2 n_{\epsilon}$')
    plt.xlabel(r'$\epsilon$ [eV]')
    plt.xscale('log')
    # plt.legend(fontsize = 15, bbox_to_anchor = [1.01, 0.5], loc = 'center left')
    plt.yscale('log')

    # ELECTRON SPECTRUM
    plt.subplot(1, 2, 2)
    ee = my_am3.get_egrid_lep() * u.eV
    plt.loglog(ee, (ee * (my_am3.get_electrons() * 1 / u.cm ** 3) * volume).to(u.eV))

    # plt.legend(loc='upper left')
    # plt.savefig('proton_synchrotron_decomposed_am3.png', dpi=300, bbox_inches = 'tight')

    return


def add_hess():
    x = [2.34283173584546e-15, 2.6079167733907346e-13, 2.15740293518768e-11, 2.148525651229302e-9, 5.432570752864949e-9,
         1.5353719098927042e-8, 5.626327301638377e-8, 1.3208741844515887e-7, 3.3398449111186517e-7,
         8.444834611523812e-7, 0.0000014733671392079496, 0.0000034589506454681524, 0.0000070003696561209326,
         0.000012675227628058634, 0.000019784826531844153, 0.000023818012772930925, 0.00019744009392108348,
         0.003987645990968316, 0.0805374442115157, 0.5343743159382559, 2.7345476523253858, 7.728500372612203,
         19.54159824144364, 57.316769663130515, 111.77505397989484, 174.47024372835168, 282.62445403136763,
         457.8235251444727, 926.5657660000949] * u.TeV
    y = [9.665440810317097e-15, 1.0137044928393754e-13, 9.70939255352522e-13, 6.2952761246427125e-12,
         8.416337272181713e-12, 9.731441211365277e-12, 8.416337272181713e-12, 6.182054511855176e-12,
         3.586572944871982e-12, 1.643476340858581e-12, 9.534767475556967e-13, 3.0948559410007366e-13,
         1.0137044928393754e-13, 2.871626747999727e-14, 9.84245911301052e-15, 6.140126695797737e-15,
         1.7752564615943787e-14, 8.079587800601521e-14, 3.0948559410007366e-13, 7.067422657716828e-13,
         1.2071833860326368e-12, 1.7042252205345882e-12, 1.9527215791252938e-12, 1.9884839566700786e-12,
         1.5993403773678951e-12, 1.2292924194672057e-12, 8.397266446387057e-13, 3.327878803776528e-13,
         7.051408377161763e-14] * u.erg / (u.cm ** 2 * u.s)
    plt.subplot(1, 2, 1)
    plt.scatter(x.to(u.eV), y.to(flux_unit), color='purple', marker='x')
    return


def compare_to_analytical(lifetime, bfield):
    bfield = bfield.to(Gauss)

    # INVERSE COMPTON SPECTRUM
    joint = JointSteadyStateFit(bvalue=bfield.value)
    joint.k10 = -3.0
    params = [.41 + np.log(lifetime / (2000 * u.yr)), 2.0, 3.0, joint.get_index(lifetime.to(u.yr)) + 1]
    plt.subplot(1, 2, 1)
    plt.loglog(joint.photon_energy, joint.model(params), color='black')

    # SYNCHROTRON RADIATION
    jee = joint.electron_energy
    dN_dE = joint.cooled_spectrum(params)
    dN_dE_inj = joint.spectrum.dn_de(jee, params[0], params[1], joint.k10, params[2])

    synchrotron_calculator = SynchrotronRadiationCalculator(bfield=bfield, electron_energy=jee)
    synch_photon_energy = synchrotron_calculator.photon_energy

    synch_real = synchrotron_calculator.photon_flux(dN_dE, joint.dist)
    plt.loglog(synch_photon_energy, synch_real, color='red')
    synch_init = synchrotron_calculator.photon_flux(dN_dE_inj, joint.dist)
    plt.loglog(synch_init, synch_real, color='green')

    # ELECTRON SPECTRA
    plt.subplot(1, 2, 2)
    plt.loglog(jee, (jee ** 2 * dN_dE).to(u.eV))
    plt.loglog(jee, (jee ** 2 * dN_dE_inj).to(u.eV))
    plt.ylim(1e58, 2e58)
    plt.xlim(1e8, 1e16)
    plt.yscale('linear')

    time = joint.times[params[-1]]
    print(time)
    print((joint.spectrum.tot_e(params[0], params[1], joint.k10, params[2]) / time).to(u.erg / u.s))
    return joint.spectrum, params


def main():
    radius = 1e2 * u.pc
    lifetime = 2000 * u.yr
    bfield = 4 * uGauss

    set_plotting_defaults()
    plt.figure(figsize=(10, 4))
    spectrum, params = compare_to_analytical(lifetime, bfield)
    my_am3 = initialize()
    my_am3 = set_switches(my_am3)
    my_am3 = set_injection_parameters(my_am3, lifetime, radius, bfield, spectrum, params)
    my_am3 = evolve_the_system(my_am3, lifetime)
    retreive_the_components(my_am3, lifetime, radius)
    add_hess()

    plt.subplot(1, 2, 2)
    plt.yscale('linear')
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    # compare_to_analytical()
    main()
