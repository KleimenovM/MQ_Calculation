import os
import pickle

import emcee
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count

import astropy.units as u
from astropy.constants import codata2010 as cst
from scipy.integrate import trapezoid

from analysis.general_spectrum_analysis import prepare_measurements, volume_approximation
from config.plotting import set_plotting_defaults, Tab10, save_figure
from config.settings import SPECTRUM_DIR, ELECTRONS_DIR, MCMC_ELECTRONS_SYNCH_ONLY, MCMC_ELECTRONS_FULL
from config.units import Franklin, Gauss, flux_unit

from src.electron_spectrum_parametrization import SpectrumParametrization
from src.likelihood_elements import log_uniform, log_likelihood_measurements


class CooledSpectrumFit:
    def __init__(self, nwalkers: int = 32, nsteps: int = 5000):
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.ndim = 3
        self.mean = np.array([0.0, 2.0, 3.0])
        self.width = np.array([5.0, 3.0, 3.0])

        self.thin = 25
        self.spectrum = SpectrumParametrization(n0=1e-26 / (u.eV * u.cm ** 3), e0=1e12 * u.eV,
                                                eta0=0.0, p0=2.0, k10=0.0, k20=3.0)

        # measurements
        data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
        names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data
        self.photon_energy, self.flux, self.flux_l, self.flux_p = prepare_measurements(e, f_cor, f_l_cor, f_p_cor)
        self.upper_indices = self.flux_p.value <= 0

        # distance and radiation area
        dist = 6.1 * u.kpc  # [kpc]
        self.area = 4 * np.pi * dist ** 2  # [kpc2]
        self.volume = volume_approximation(self.photon_energy).to(u.cm ** 3)  # [pc3]

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

    def cooled_spectrum(self, parameters):
        return 0.0 * 1 / (u.eV * u.cm ** 3)

    def model(self, parameters):
        return 0.0 * flux_unit

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        model = self.model(theta)
        return lp + log_likelihood_measurements(self.flux, self.flux_l, self.flux_p, self.upper_indices, model)

    def run(self):
        return


class SynchrotronCooledSpectrumFit(CooledSpectrumFit):
    def __init__(self, nwalkers: int = 32, nsteps: int = 5000):
        super().__init__(nwalkers, nsteps)

        # MCMC parameters
        self.ndim = 3  # two spectral parameters (norm + pwl) and b2t
        # source parameters   eta gamma  B^2t
        self.mean = np.array([0.0, 2.0, 3.0])  # [DL]
        self.width = np.array([5.0, 3.0, 3.0])  # [DL]
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)  # [DL]

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

    def cooled_spectrum(self, parameters):
        eta, gamma, b2t = parameters[0], parameters[1], 10**parameters[2]

        # backpropagated energies
        allowed = self.sharp_1Gauss * b2t * self.electron_energy < 1
        allowed_energy = self.electron_energy[allowed]
        modified_energy = allowed_energy / (1 - self.sharp_1Gauss * b2t * allowed_energy)  # [eV]

        # electron number density after cooling
        dn_de_cooled = np.zeros_like(self.electron_energy.value) * (1 / (u.eV * u.cm ** 3))
        dn_de_cooled[allowed] = (self.spectrum.dn_de(modified_energy, eta, gamma, -3.0, 4.0) /
                                 (1 - self.sharp_1Gauss * b2t * allowed_energy) ** 2)  # [eV-1 cm-3]
        return dn_de_cooled

    def model(self, parameters):
        # electron density after cooling
        dn_de_cooled = self.cooled_spectrum(parameters)

        # number of photons homogenously radiated per unit volume of the nebula (dN/dE.dt.dV)
        photon_spectrum = trapezoid(self.radiation_matrix.T * dn_de_cooled, self.electron_energy)  # [eV-1 s-1 cm-3]

        # photon flux from the whole source
        flux = self.photon_energy ** 2 * photon_spectrum * self.volume / self.area
        return flux.to(flux_unit)

    def run(self):
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability, pool=pool)
            sampler.run_mcmc(self.start, self.nsteps, progress=True)
        return sampler.get_chain(discard=int(0.2 * self.nsteps), thin=self.thin, flat=True)


class JointCooledSpectrumFit(CooledSpectrumFit):
    def __init__(self, nsteps=2000, nwalkers=32, bvalue: float = 1e-6):
        super().__init__(nwalkers, nsteps)

        # MCMC parameters
        self.ndim = 3  # two spectral parameters (norm + pwl) and b2t
        # source parameters   eta gamma  B^2t
        self.mean = np.array([0.0, 2.0, 5000])  # [DL]
        self.width = np.array([5.0, 3.0, 4999])  # [DL]
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)  # [DL]

        # electron radiation properties
        data = pickle.load(open(os.path.join(ELECTRONS_DIR, "spectrum_interpolated.pck"), "rb"))
        self.lg_photon_energy = np.log10(self.photon_energy.to(u.eV).value)
        interpolator = data[2]

        # electron cooling precomputed
        times, energies, self.modulation_coefficient = (
            pickle.load(open(os.path.join(ELECTRONS_DIR, f"joint_cooling_{bvalue * 1e6:.1f}.pck"), "rb")))
        self.times = times * u.yr
        self.energies = energies * u.eV
        self.electron_energy = self.energies[:, 0]
        lg_electron_energy = np.log10(self.electron_energy.value)  # initial energy

        # radiation matrix
        lg_xy, lg_yx = np.meshgrid(lg_electron_energy, self.lg_photon_energy, indexing='ij')
        self.radiation_matrix = 10 ** interpolator((lg_xy, lg_yx)) * (1 / (u.eV * u.s))

    def cooled_spectrum(self, parameters):
        eta, gamma, time_index = parameters[0], parameters[1], np.clip(int(parameters[2]), 0, len(self.times) - 1)
        e_time = np.clip(self.energies[:, time_index].value, 1e-10, 1e35) * u.eV
        dn_de1 = self.spectrum.dn_de(e_time, eta=eta, p=gamma, k1=-5.0, k2=4.0)
        return dn_de1 * self.modulation_coefficient[:, time_index]

    def model(self, parameters):
        # electron density after cooling
        dn_de_cooled = self.cooled_spectrum(parameters)

        # number of photons homogenously radiated per unit volume of the nebula (dN/dE.dt.dV)
        photon_spectrum = trapezoid(self.radiation_matrix.T * dn_de_cooled, self.electron_energy)

        # photon flux from the whole source
        flux = self.photon_energy ** 2 * photon_spectrum * self.volume / self.area
        return flux.to(flux_unit)

    def run(self):
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability)
        sampler.run_mcmc(self.start, self.nsteps, progress=True)
        return sampler.get_chain(discard=int(0.2 * self.nsteps), thin=self.thin, flat=True)


def electron_synchrotron_only(nsteps=2000, nwalkers=32):
    synch = SynchrotronCooledSpectrumFit(nsteps=nsteps, nwalkers=nwalkers)
    result = synch.run()
    with open(os.path.join(MCMC_ELECTRONS_SYNCH_ONLY,
                           f"synch_{nsteps}n_{nwalkers}w_3_param_chain.pck"), "wb") as file_open:
        pickle.dump([result, synch], file_open)
    return


def electron_joint_cooling(nsteps=2000, nwalkers=32, bvalue: float = 1e-6):
    joint = JointCooledSpectrumFit(nsteps=nsteps, nwalkers=nwalkers)
    result = joint.run()
    with open(os.path.join(MCMC_ELECTRONS_FULL,
                           f"joint_{nsteps}n_{nwalkers}w_{bvalue * 1e6:.1f}uG.pck"), "wb") as file_open:
        pickle.dump([result, joint], file_open)
    return


if __name__ == '__main__':
    n = 2000
    nw = 64
    bvalue = 4e-6
    # electron_synchrotron_only(nsteps=n, nwalkers=nw)
    electron_joint_cooling(nsteps=n, nwalkers=nw, bvalue=bvalue)
