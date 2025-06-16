import os
import pickle
from multiprocessing import Pool

import emcee
import numpy as np
import astropy.units as u
from aafragpy import get_cross_section, get_cross_section_Kamae2006, get_spectrum

import astropy.constants.codata2010 as cst
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from analysis_electrons.general_spectrum_analysis import prepare_measurements
from config.settings import SPECTRUM_DIR, ELECTRONS_DIR, INVERSE_COMPTON_DIR, MCMC_PROTONS
from config.units import Gauss, Franklin
from src.electron_spectrum_parametrization import SpectrumParametrization
from src.electron_timescales import synchrotron_timescale
from src.likelihood_elements import log_uniform, log_likelihood_measurements


class ProtonElectronSpectrumFit:
    def __init__(self, nwalkers: int = 32, nsteps: int = 5000, bfield =1e-6 * Gauss):
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.ndim = 4  # four spectral parameters
        # source parameters   eta gamma  k1
        self.mean = np.array([0.0, 2.0, 5.0, 5500])  # [DL]
        self.width = np.array([10.0, 3.0, 5.0, 4499])  # [DL]
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)  # [DL]

        self.thin = 25

        self.k10 = -10.0

        self.bfield = bfield
        self.spectrum = SpectrumParametrization(n0=1e36 / u.eV, e0=1e12 * u.eV,
                                                eta0=0.0, p0=2.0, k10=self.k10, k20=3.0)

        # measurements
        data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
        names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data
        self.photon_energy, self.flux, self.flux_l, self.flux_p = prepare_measurements(e, f_cor, f_l_cor, f_p_cor)
        self.upper_indices = self.flux_p.value <= 0

        # cold proton density
        self.density = 1 * u.cm ** (-3)

        # protons
        self.proton_energy = np.logspace(9, 19, 500) * u.eV

        # protons -> gamma
        (self.e_prim, self.above_cutoff, self.below_cutoff,
         self.e_sec_phot,
         self.he_cs_matrix, self.le_cs_matrix) = self.get_cs_matrices()

        # p -> e cross-section matrix
        # secondary electrons
        self.electron_energy = np.logspace(9, 19, 500) * u.eV
        self.lg_electron_energy = np.log10(self.electron_energy.value)
        self.e_sec_els = self.electron_energy.to(u.GeV).value

        print('cs electron/positron calculation')
        self.el_cs_matrix = get_cross_section(secondary='el', primary_target='p-p',
                                              E_primaries=self.e_prim[self.above_cutoff],
                                              E_secondaries=self.e_sec_els)
        self.pos_cs_matrix = get_cross_section(secondary='posi', primary_target='p-p',
                                               E_primaries=self.e_prim[self.above_cutoff],
                                               E_secondaries=self.e_sec_els)

        # distance and radiation area
        dist = 6.1 * u.kpc  # [kpc]
        self.area = 4 * np.pi * dist ** 2  # [kpc2]

        # e -> gamma
        # IC radiation properties
        self.electron_energy, self.radiation_matrix = self.set_radiation_matrix()

        # synchrotron radiation parameters
        sin2_avg = 2 / 3
        e = cst.e.gauss.value * Franklin
        self.sharp = ((2 * e ** 4 / (3 * cst.m_e ** 4 * cst.c ** 7) * self.bfield ** 2 * sin2_avg).to(
            u.eV ** (-1) * u.yr ** (-1)))

        t_min = synchrotron_timescale(energy=1e19 * u.eV, bfield=self.bfield).to(u.yr)
        lg_t_min = np.log10(t_min.value)
        self.times = np.logspace(lg_t_min, 6, 10000) * u.yr  # [G2 yr]

        # transport solutions
        e_t, t_e = np.meshgrid(self.electron_energy, self.times, indexing='ij')
        allowed = self.sharp * t_e * e_t < 1
        allowed_energy = e_t[allowed]

        # energy backpropagation
        self.e_mod = np.ones_like(e_t) * 1e35  # [eV], very high energy instead of non-allowed energies
        self.e_mod[allowed] = allowed_energy / (1 - self.sharp * t_e * e_t)[allowed]
        self.lg_e_mod = np.log10(self.e_mod.value)

        # modulation coefficient
        self.phi_matrix = np.zeros(e_t.shape)  # [DL], modulation 0 for non-allowed energies
        self.phi_matrix[allowed] = ((1 - self.sharp * t_e * e_t)[allowed]) ** (-2)

    def get_cs_matrices(self, primary_energy=None,
                        secondary_energy=None):
        if primary_energy is None:
            primary_energy = self.proton_energy

        e_prim = primary_energy.to(u.GeV).value

        af_cutoff = 4.0  # [GeV], aafrag high-energy cutoff
        above_cutoff = e_prim > af_cutoff
        below_cutoff = e_prim <= af_cutoff

        if secondary_energy is None:
            secondary_energy = self.photon_energy
        e_sec = secondary_energy.to(u.GeV).value

        # cross-section matrix
        he_cs_matrix = get_cross_section(secondary='gam', primary_target='p-p',
                                         E_primaries=e_prim[above_cutoff],
                                         E_secondaries=e_sec)
        le_cs_matrix = get_cross_section_Kamae2006('gam',
                                                   E_primaries=e_prim[below_cutoff],
                                                   E_secondaries=e_sec)
        return e_prim, above_cutoff, below_cutoff, e_sec, he_cs_matrix, le_cs_matrix

    def set_radiation_matrix(self, electron_energy=None, take_each: int = 4,
                             photon_energy=None):
        # electron radiation properties
        data = pickle.load(open(os.path.join(INVERSE_COMPTON_DIR, "IC_Total.pck"), "rb"))
        if photon_energy is None:
            photon_energy = self.photon_energy
        lg_photon_energy = np.log10(photon_energy.to(u.eV).value)

        interpolator = data[2]

        if electron_energy is None:
            lg_electron_energy = data[0][::take_each]
            electron_energy = 10 ** lg_electron_energy * u.eV
        else:
            lg_electron_energy = np.log10(electron_energy.value)

        lg_xy, lg_yx = np.meshgrid(lg_electron_energy, lg_photon_energy, indexing='ij')
        radiation_matrix = 10 ** interpolator((lg_xy, lg_yx)) * (1 / (u.eV * u.s))

        return electron_energy, radiation_matrix

    def secondary_emission(self, parameters):
        eta, gamma, k2, time_index = parameters[0], parameters[1], parameters[2], int(parameters[-1])
        dN_dEp = self.spectrum.dn_de(self.proton_energy[self.above_cutoff], eta, gamma, self.k10, k2).value
        spec_unit = u.mbarn / u.eV
        electron_spectrum = get_spectrum(energy_primary=self.e_prim[self.above_cutoff],
                                         energy_secondary=self.e_sec_els,
                                         cs_matrix=self.el_cs_matrix[0] + self.pos_cs_matrix[0],
                                         prim_spectrum=dN_dEp) * spec_unit

        electron_emission = (electron_spectrum * self.density * cst.c).to(1 / (u.eV * u.s))
        return electron_emission

    def secondary_electrons(self, parameters):
        eta, gamma, k2, time_index = parameters[0], parameters[1], parameters[2], int(parameters[-1])
        electron_emission = self.secondary_emission(parameters)

        lg_el_emis = np.clip(np.log10(electron_emission.value), a_min=-20, a_max=1e5)
        electron_emission_interpolation = interp1d(self.lg_electron_energy, lg_el_emis,
                                                   bounds_error=False, fill_value=-32)

        # backpropagated energies
        t1 = self.times[:time_index + 1]
        lg_e1 = self.lg_e_mod[:, :time_index + 1]

        # electron number density after cooling
        q_matrix = 10**electron_emission_interpolation(lg_e1) * 1/(u.eV * u.s)
        phi_matrix = self.phi_matrix[:, :time_index + 1]
        dN_dE = trapezoid(q_matrix * phi_matrix, t1)
        return dN_dE.to(1/u.eV)

    def log_prior(self, theta):
        """
        Calculate log prior
        :param theta:
        :return:
        """
        result = 0
        for i in range(self.ndim):
            result += log_uniform(theta[i], self.mean[i], self.width[i])
        return result

    def proton_photons(self, parameters):
        # number of photons homogenously radiated per unit volume of the nebula (dN/dE.dt.dV)
        eta, gamma, k2, time = parameters[0], parameters[1], parameters[2], int(parameters[-1])
        dN_dEp = self.spectrum.dn_de(self.proton_energy, eta, gamma, self.k10, k2).value
        spec_unit = u.mbarn / u.eV

        photon_he_spectrum = get_spectrum(energy_primary=self.e_prim[self.above_cutoff],
                                          energy_secondary=self.e_sec_phot,
                                          cs_matrix=self.he_cs_matrix[0],
                                          prim_spectrum=dN_dEp[self.above_cutoff]) * spec_unit

        photon_le_spectrum = get_spectrum(energy_primary=self.e_prim[self.below_cutoff],
                                          energy_secondary=self.e_sec_phot,
                                          cs_matrix=self.le_cs_matrix[0],
                                          prim_spectrum=dN_dEp[self.below_cutoff]) * spec_unit

        flux = self.photon_energy ** 2 * (photon_he_spectrum + photon_le_spectrum) * self.density * cst.c / self.area
        return flux.to(u.erg / (u.s * u.cm ** 2))

    def electron_photons(self, parameters):
        electron_spectrum = self.secondary_electrons(parameters)
        photon_el_spectrum = trapezoid(self.radiation_matrix.T * electron_spectrum, self.electron_energy)

        flux = self.photon_energy ** 2 * photon_el_spectrum / self.area
        return flux.to(u.erg / (u.s * u.cm ** 2))

    def model(self, parameters):
        return self.proton_photons(parameters) + self.electron_photons(parameters)

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        model = self.model(theta)
        return lp + log_likelihood_measurements(self.flux, self.flux_l, self.flux_p, self.upper_indices, model)

    def run(self):
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability)
        sampler.run_mcmc(self.start, self.nsteps, progress=True)
        return sampler.get_chain(discard=int(0.2 * self.nsteps), thin=self.thin, flat=True)


def run_protons(nsteps=2000, nwalkers=32, bfield: float = 1e-6):
    pesf = ProtonElectronSpectrumFit(nsteps=nsteps, nwalkers=nwalkers, bfield=bfield)
    pesf.k10 = -10.0
    result = pesf.run()
    with open(os.path.join(MCMC_PROTONS,
                           f"protons_{nsteps}n_{nwalkers}w_{bfield.value * 1e6:.1f}uG.pck"), "wb") as file_open:
        pickle.dump([result, pesf], file_open)
    return


if __name__ == '__main__':
    n = 400
    nw = 8
    run_protons(nsteps=n, nwalkers=nw, bfield=10e-6 * Gauss)
