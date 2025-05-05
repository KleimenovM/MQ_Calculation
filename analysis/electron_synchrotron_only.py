import os
import pickle

import emcee
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count

import astropy.units as u
from astropy.constants import codata2010 as cst
from scipy.integrate import trapezoid

from config.plotting import set_plotting_defaults, Tab10
from config.settings import SPECTRUM_DIR, ELECTRONS_DIR, MCMC_ELECTRONS_SYNCH_ONLY
from config.units import Franklin, Gauss, flux_unit

from src.electron_spectrum_parametrization import SpectrumParametrization
from src.likelihood_elements import log_uniform, log_likelihood_measurements


class SynchrotronOnly:
    def __init__(self, nwalkers: int = 32, nsteps: int = 5000):
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.ndim = 5  # four spectral parameters + b2t
        # source parameters   eta gamma  k1   k2   B^2t
        self.mean = np.array([0.0, 2.0, 0.0, 5.0, 3.0])  # [DL]
        self.width = np.array([5.0, 3.0, 3.0, 5.0, 3.0])  # [DL]
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)   # [DL]

        self.thin = 25

        self.spectrum = SpectrumParametrization(n0=1e-27 / (u.eV * u.cm ** 3), e0=1e12 * u.eV,
                                                eta0=0.0, p0=2.0, k10=0.0, k20=3.0)

        # measurements
        data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
        names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data
        self.photon_energy, self.flux, self.flux_l, self.flux_p = self.prepare_measurements(e, f_cor, f_l_cor, f_p_cor)
        self.upper_indices = self.flux_p.value <= 0

        # electron radiation properties
        data = pickle.load(open(os.path.join(ELECTRONS_DIR, "spectrum_interpolated.pck"), "rb"))
        lg_photon_energy = np.log10(self.photon_energy.to(u.eV).value)
        lg_electron_energy, interpolator = data[0], data[2]
        self.electron_energy = 10 ** lg_electron_energy * u.eV

        lg_xy, lg_yx = np.meshgrid(lg_electron_energy, lg_photon_energy, indexing='ij')
        self.radiation_matrix = 10 ** interpolator((lg_xy, lg_yx)) * (1 / (u.eV * u.s))

        # synchrotron radiation parameters
        sin2_avg = 2 / 3
        e = cst.e.gauss.value * Franklin
        self.sharp_1Gauss = ((2 * e ** 4 / (3 * cst.m_e ** 4 * cst.c ** 7) * (1e-6 * Gauss) ** 2 * sin2_avg).to(
            u.eV ** (-1) * u.yr ** (-1)) * u.yr)

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
        theta = parameters[:-1]  # [DL]
        b2t = 10 ** parameters[-1]  # [DL]

        # backpropagated energies
        allowed = self.sharp_1Gauss * b2t * self.electron_energy < 1
        allowed_energy = self.electron_energy[allowed]
        modified_energy = allowed_energy / (1 - self.sharp_1Gauss * b2t * allowed_energy)  # [eV]

        # electron number density after cooling
        dn_de_cooled = np.zeros_like(self.electron_energy.value) * (1 / (u.eV * u.cm ** 3))
        dn_de_cooled[allowed] = (self.spectrum.dn_de(modified_energy, *theta) /
                                 (1 - self.sharp_1Gauss * b2t * allowed_energy) ** 2)  # [eV-1 cm-3]

        # number of photons homogenously radiated per unit volume of the nebula (dN/dE.dt.dV)
        photon_spectrum = trapezoid(self.radiation_matrix.T * dn_de_cooled, self.electron_energy)  # [eV-1 s-1 cm-3]

        # photon flux from the whole source
        flux = self.photon_energy ** 2 * photon_spectrum * self.volume / self.area
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


def electron_synchrotron_only(nsteps=2000, nwalkers=32):
    synch = SynchrotronOnly(nsteps=nsteps, nwalkers=nwalkers)
    result = synch.run()

    pickle.dump([result, synch],
                open(os.path.join(MCMC_ELECTRONS_SYNCH_ONLY, f"electrons_{nsteps}n_{nwalkers}w.pck"), "wb"))

    return


def plot_mcmc_results(nsteps=2000, nwalkers=32):
    data = pickle.load(open(os.path.join(MCMC_ELECTRONS_SYNCH_ONLY, f"electrons_{nsteps}n_{nwalkers}w.pck"), "rb"))
    result, synch = data[0], data[1]

    set_plotting_defaults()
    mean_result = np.mean(result, axis=0)

    print(mean_result)

    electron_number_density = synch.spectrum.tot_n(*mean_result[:-1])
    electron_energy_density = synch.spectrum.tot_e(*mean_result[:-1])

    print(electron_number_density, electron_energy_density)

    data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
    names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data

    set_plotting_defaults()

    for i, name in enumerate(names):
        plt.errorbar(e[i], f_cor[i], xerr=[e_l[i], e_p[i]], yerr=[f_l_cor[i], f_p_cor[i]],
                     fmt='o', linestyle='None', uplims=f_p_cor[i] <= 0,
                     color=Tab10[i], label=f' ')

    for res in result[::nsteps // 200]:
        plt.plot(synch.photon_energy, synch.model(res), alpha=.02, color='blue')

    plt.xscale('log')
    plt.xlim(1e9, 1e15)
    plt.xlabel("Energy, eV")

    plt.yscale('log')
    plt.ylim(1e-13, 1e-10)
    plt.ylabel(r"Flux, $\mathrm{erg~cm^{-2}~s^{-1}}$")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    n = 2000
    nw = 64
    # electron_synchrotron_only(nsteps=n, nwalkers=nw)
    plot_mcmc_results(n, nw)
