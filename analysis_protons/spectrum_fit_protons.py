import os
import pickle

import emcee
import numpy as np

from multiprocessing import Pool, cpu_count

from aafragpy import get_cross_section, get_spectrum, get_cross_section_Kamae2006

import astropy.units as u
from astropy.constants import codata2010 as cst
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from analysis_electrons.general_spectrum_analysis import volume_approximation, prepare_measurements
from config.settings import SPECTRUM_DIR, MCMC_PROTONS, ELECTRONS_DIR
from config.units import Franklin, Gauss

from src.electron_spectrum_parametrization import SpectrumParametrization
from src.electron_timescales import synchrotron_timescale
from src.likelihood_elements import log_uniform, log_likelihood_measurements


class OnlyProtonSpectrumFit:
    def __init__(self, nwalkers: int = 32, nsteps: int = 5000):
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.ndim = 3  # three spectral parameters
        # source parameters   eta gamma  k1
        self.mean = np.array([0.0, 2.0, 5.0])  # [DL]
        self.width = np.array([5.0, 3.0, 5.0])  # [DL]
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)  # [DL]

        self.thin = 25

        self.spectrum = SpectrumParametrization(n0=1e36 / u.eV, e0=1e12 * u.eV,
                                                eta0=0.0, p0=2.0, k10=-10.0, k20=3.0)

        # measurements
        data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
        names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data
        self.photon_energy, self.flux, self.flux_l, self.flux_p = prepare_measurements(e, f_cor, f_l_cor, f_p_cor)
        self.upper_indices = self.flux_p.value <= 0

        self.density = 1 * u.cm ** (-3)

        # proton, primary and secondary energies for aafragpy
        self.proton_energy = np.logspace(9, 19, 500) * u.eV

        (self.e_prim, self.above_cutoff,self.below_cutoff,
         self.e_sec,
         self.he_cs_matrix, self.le_cs_matrix) = self.get_cs_matrices()

        # distance and radiation area
        dist = 6.1 * u.kpc  # [kpc]
        self.area = 4 * np.pi * dist ** 2  # [kpc2]
        return

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
                                         E_primaries=self.e_prim[self.above_cutoff],
                                         E_secondaries=self.e_sec)
        le_cs_matrix = get_cross_section_Kamae2006('gam',
                                                   E_primaries=self.e_prim[self.below_cutoff],
                                                   E_secondaries=self.e_sec)
        return e_prim, above_cutoff, below_cutoff, e_sec, he_cs_matrix, le_cs_matrix

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

    def model(self, parameters):
        # number of photons homogenously radiated per unit volume of the nebula (dN/dE.dt.dV)
        eta, gamma, k2 = parameters[0], parameters[1], parameters[2]
        dN_dEp = self.spectrum.dn_de(self.proton_energy, eta, gamma, -10, k2).value
        spec_unit = u.mbarn / u.eV
        photon_he_spectrum = get_spectrum(energy_primary=self.e_prim[self.above_cutoff],
                                          energy_secondary=self.e_sec,
                                          cs_matrix=self.he_cs_matrix[0],
                                          prim_spectrum=dN_dEp[self.above_cutoff]) * spec_unit
        photon_le_spectrum = get_spectrum(energy_primary=self.e_prim[self.below_cutoff],
                                          energy_secondary=self.e_sec,
                                          cs_matrix=self.le_cs_matrix[0],
                                          prim_spectrum=dN_dEp[self.below_cutoff]) * spec_unit
        photon_spectrum = photon_he_spectrum + photon_le_spectrum

        # photon flux from the whole source
        flux = self.photon_energy ** 2 * photon_spectrum / self.area * self.density * cst.c
        return flux.to(u.erg / (u.s * u.cm ** 2))

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        model = self.model(theta)
        return lp + log_likelihood_measurements(self.flux, self.flux_l, self.flux_p, self.upper_indices, model)

    def run(self):
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability, pool=pool)
            sampler.run_mcmc(self.start, self.nsteps, progress=True)
        return sampler.get_chain(discard=int(0.2 * self.nsteps), thin=self.thin, flat=True)


class ProtonSecondaryElectrons:
    def __init__(self, proton_energy, sec_energy):
        # proton, primary and secondary energies for aafragpy
        self.proton_energy = proton_energy
        self.e_prim = self.proton_energy.to(u.GeV).value

        af_cutoff = 4.0  # [GeV], aafrag high-energy cutoff
        self.above_cutoff = self.e_prim > af_cutoff
        self.below_cutoff = self.e_prim <= af_cutoff

        self.sec_energy = sec_energy
        self.e_sec = self.sec_energy.to(u.GeV).value

        self.density = 1 * u.cm ** (-3)

        self.spectrum = SpectrumParametrization(n0=1e36 / u.eV, e0=1e12 * u.eV,
                                                eta0=0.0, p0=2.0, k10=-10.0, k20=3.0)

        # cross-section matrix
        print('cs calculation')
        self.el_cs_matrix = get_cross_section(secondary='el', primary_target='p-p',
                                              E_primaries=self.e_prim[self.above_cutoff],
                                              E_secondaries=self.e_sec)
        self.pos_cs_matrix = get_cross_section(secondary='posi', primary_target='p-p',
                                               E_primaries=self.e_prim[self.above_cutoff],
                                               E_secondaries=self.e_sec)
        print('ready')

        # distance and radiation area
        dist = 6.1 * u.kpc  # [kpc]
        self.area = 4 * np.pi * dist ** 2  # [kpc2]
        return

    def prep_aafrag_spectrum(self, parameters):
        eta, gamma, k2 = parameters[0], parameters[1], parameters[2]
        return self.spectrum.dn_de(self.proton_energy, eta, gamma, -10, k2).value

    def sec_density(self, parameters, particle: str):
        if particle not in ['el', 'posi']:
            raise ValueError(f"Particle {particle} not expected. Only 'el' and 'pos' are allowed.")
        if particle == 'el':
            csm = self.el_cs_matrix
        else:
            csm = self.pos_cs_matrix

        spec_unit = u.mbarn / u.eV

        dN_dEp = self.prep_aafrag_spectrum(parameters)
        spectrum = get_spectrum(energy_primary=self.e_prim[self.above_cutoff],
                                energy_secondary=self.e_sec,
                                cs_matrix=csm[0],
                                prim_spectrum=dN_dEp[self.above_cutoff]) * spec_unit

        density = spectrum * self.density * cst.c
        return density.to(1/(u.eV * u.s))

    def sec_flux(self, sec_density):
        flux = self.sec_energy ** 2 * sec_density / self.area
        return flux.to(u.erg / (u.s * u.cm ** 2))


def proton_fit(nsteps=2000, nwalkers=32):
    proton_spectrum_fit = OnlyProtonSpectrumFit(nsteps=nsteps, nwalkers=nwalkers)
    result = proton_spectrum_fit.run()

    pickle.dump([result, proton_spectrum_fit],
                open(os.path.join(MCMC_PROTONS, f"protons_{nsteps}n_{nwalkers}w.pck"), "wb"))

    return


if __name__ == '__main__':
    n = 2000
    nw = 64
    proton_fit(nsteps=n, nwalkers=nw)
