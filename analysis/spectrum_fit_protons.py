import os
import pickle

import emcee
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count

from aafragpy import get_cross_section, get_spectrum

import astropy.units as u
from astropy.constants import codata2010 as cst
from scipy.integrate import trapezoid

from config.plotting import set_plotting_defaults, Tab10, save_figure
from config.settings import SPECTRUM_DIR, MCMC_PROTONS
from config.units import Franklin, Gauss, flux_unit

from src.electron_spectrum_parametrization import SpectrumParametrization
from src.likelihood_elements import log_uniform, log_likelihood_measurements


class ProtonSpectrumFit:
    def __init__(self, nwalkers: int = 32, nsteps: int = 5000):
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.ndim = 4  # four spectral parameters + b2t
        # source parameters   eta gamma  k1   k2   B^2t
        self.mean = np.array([0.0, 2.0, 0.0, 5.0])  # [DL]
        self.width = np.array([5.0, 3.0, 10.0, 5.0])  # [DL]
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)   # [DL]

        self.thin = 25

        self.spectrum = SpectrumParametrization(n0=1e-23 / (u.eV * u.cm ** 3), e0=1e12 * u.eV,
                                                eta0=0.0, p0=2.0, k10=0.0, k20=3.0)

        # measurements
        data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
        names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data
        self.photon_energy, self.flux, self.flux_l, self.flux_p = self.prepare_measurements(e, f_cor, f_l_cor, f_p_cor)
        self.upper_indices = self.flux_p.value <= 0

        # proton, primary and secondary energies for aafragpy
        self.proton_energy = np.logspace(12, 19, 500) * u.eV
        self.primary_energies = self.proton_energy.to(u.GeV).value
        self.secondary_energies = self.photon_energy.to(u.GeV).value

        self.density = 1 * u.cm**(-3)

        # cross-section matrix
        self.cs_matrix = get_cross_section(secondary='gam', primary_target='p-p',
                                           E_primaries=self.primary_energies, E_secondaries=self.secondary_energies)

        # distance and radiation area
        dist = 6.6 * u.kpc  # [kpc]
        self.area = 4 * np.pi * dist ** 2  # [kpc2]
        self.volume = self.volume_approximation(self.photon_energy).to(u.cm ** 3)  # [pc3]
        return

    @staticmethod
    def volume_approximation(energy):
        # volume (define by Kolmogorov diffusion)
        width_length_ratio = 0.2
        size_1TeV = 100 * u.pc
        size = size_1TeV * (energy / u.TeV) ** (1 / 6)
        return size ** 3 * width_length_ratio ** 2

    @staticmethod
    def prepare_measurements(e, f_cor, f_l_cor, f_p_cor):
        energy = []
        flux, flux_l, flux_p = [], [], []

        for i in range(len(e)):
            for j in range(len(e[i])):
                energy.append(e[i][j].value)
                flux.append(f_cor[i][j].value)
                flux_l.append(f_l_cor[i][j].value)
                flux_p.append(f_p_cor[i][j].value)

        e = np.array(energy) * u.eV
        f, f_l, f_p = np.array(flux) * flux_unit, np.array(flux_l) * flux_unit, np.array(flux_p) * flux_unit

        indices = np.argsort(e)
        return e[indices], f[indices], f_l[indices], f_p[indices]

    def log_prior(self, theta):
        """
        Calculate log prior
        :param theta:
        :return:
        """
        if theta[2] > theta[3]:
            return -np.inf
        result = 0
        for i in range(self.ndim):
            result += log_uniform(theta[i], self.mean[i], self.width[i])
        return result

    def model(self, parameters):
        # number of photons homogenously radiated per unit volume of the nebula (dN/dE.dt.dV)
        dN_dEp = self.spectrum.dn_de(self.proton_energy, *parameters).value
        photon_spectrum = get_spectrum(energy_primary=self.primary_energies, energy_secondary=self.secondary_energies,
                                       cs_matrix=self.cs_matrix[0], prim_spectrum=dN_dEp) * u.mbarn / (u.eV * u.cm ** 3)

        # photon flux from the whole source
        flux = self.photon_energy ** 2 * photon_spectrum * self.volume / self.area * self.density * cst.c
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


def proton_fit(nsteps=2000, nwalkers=32):
    proton_spectrum_fit = ProtonSpectrumFit(nsteps=nsteps, nwalkers=nwalkers)
    result = proton_spectrum_fit.run()

    pickle.dump([result, proton_spectrum_fit],
                open(os.path.join(MCMC_PROTONS, f"protons_{nsteps}n_{nwalkers}w.pck"), "wb"))

    return


if __name__ == '__main__':
    n = 2000
    nw = 64
    proton_fit(nsteps=n, nwalkers=nw)
