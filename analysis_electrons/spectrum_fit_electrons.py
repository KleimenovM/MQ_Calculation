import os
import pickle

import emcee
import numpy as np

from multiprocessing import Pool, cpu_count

import astropy.units as u
from astropy.constants import codata2010 as cst
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

from analysis_electrons.general_spectrum_analysis import prepare_measurements
from config.settings import SPECTRUM_DIR, ELECTRONS_DIR, MCMC_ELECTRONS_SYNCH_ONLY, MCMC_ELECTRONS_JOINT, \
    JOINT_COOLING_DIR, MCMC_ELECTRONS_STEADY_JOINT, MCMC_ELECTRONS_SYNCH_STEADY, INVERSE_COMPTON_DIR
from config.units import Franklin, Gauss, flux_unit

from src.electron_spectrum_parametrization import SpectrumParametrization
from src.electron_timescales import synchrotron_timescale
from src.likelihood_elements import log_uniform, log_likelihood_measurements


class CooledSpectrumFit:
    def __init__(self, nwalkers: int = 32, nsteps: int = 5000):
        self.nwalkers = nwalkers
        self.nsteps = nsteps

        self.ndim = 0
        self.mean = np.array([])
        self.width = np.array([])

        self.start = self.mean

        self.thin = 25

        self.k10 = -5.0
        self.k20 = 6.0
        self.spectrum = SpectrumParametrization(n0=1e34 / u.eV,  # number of ejected particles dN/dE
                                                e0=1e12 * u.eV,  # initial reference energy (1 TeV)
                                                eta0=0.0, p0=2.0, k10=0.0, k20=6.0)

        # measurements
        data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
        names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data
        self.photon_energy, self.flux, self.flux_l, self.flux_p = prepare_measurements(e, f_cor, f_l_cor, f_p_cor)
        self.upper_indices = self.flux_p.value <= 0

        # distance and radiation area
        dist = 6.1 * u.kpc  # [kpc]
        self.area = 4 * np.pi * dist ** 2  # [kpc2]

        self.electron_energy = .0 * u.eV
        self.radiation_matrix = .0 * 1 / (u.eV * u.s)

    def set_radiation_matrix(self, electron_energy=None, take_each: int = 1,
                             photon_energy=None):
        # electron radiation properties
        data = pickle.load(open(os.path.join(INVERSE_COMPTON_DIR, "IC_Total.pck"), "rb"))
        if photon_energy is None:
            lg_photon_energy = np.log10(self.photon_energy.to(u.eV).value)
        else:
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
        return 0.0 * 1 / u.eV

    def model(self, parameters):
        # electron density after cooling
        dN_dE_cooled = self.cooled_spectrum(parameters)

        # number of photons homogenously radiated from the nebula (dN/dE.dt)
        photon_spectrum = trapezoid(self.radiation_matrix.T * dN_dE_cooled, self.electron_energy)

        # photon flux from the whole source
        flux = self.photon_energy ** 2 * photon_spectrum / self.area
        return flux.to(flux_unit)

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


class SynchrotronCooledSpectrumFit(CooledSpectrumFit):
    def __init__(self, nwalkers: int = 32, nsteps: int = 5000):
        super().__init__(nwalkers, nsteps)

        # MCMC parameters
        self.ndim = 4  # two spectral parameters (norm + pwl) and b2t
        # source parameters   eta gamma  lg(B^2t)
        self.mean = np.array([0.0, 2.0, 0.0, 3])  # [DL]
        self.width = np.array([5.0, 3.0, 3.0, 3])  # [DL]
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)  # [DL]

        # synchrotron radiation parameters
        sin2_avg = 2 / 3
        e = cst.e.gauss.value * Franklin
        self.sharp_1Gauss = ((2 * e ** 4 / (3 * cst.m_e ** 4 * cst.c ** 7) * (1e-6 * Gauss) ** 2 * sin2_avg).to(
            u.eV ** (-1) * u.yr ** (-1)) * u.yr)

        self.electron_energy, self.radiation_matrix = self.set_radiation_matrix()

    def cooled_spectrum(self, parameters):
        eta, gamma, k1, b2t = parameters[0], parameters[1], parameters[2], 10 ** parameters[3]

        # backpropagated energies
        allowed = self.sharp_1Gauss * b2t * self.electron_energy < 1
        allowed_energy = self.electron_energy[allowed]
        modified_energy = allowed_energy / (1 - self.sharp_1Gauss * b2t * allowed_energy)  # [eV]

        # electron number density after cooling
        dN_de_cooled = np.zeros_like(self.electron_energy.value) * (1 / u.eV)
        dN_de_cooled[allowed] = (self.spectrum.dn_de(modified_energy, eta, gamma, self.k10, self.k20) /
                                 (1 - self.sharp_1Gauss * b2t * allowed_energy) ** 2)
        return dN_de_cooled


class SynchrotronSteadyStateFit(CooledSpectrumFit):
    def __init__(self, nwalkers: int = 32, nsteps: int = 5000):
        super().__init__(nwalkers, nsteps)

        # MCMC parameters
        self.ndim = 4  # four spectral parameters (norm, pwl, k1, k2) and b2t
        # source parameters   eta gamma k2   B^2t
        self.mean = np.array([0.0, 2.0, 3.0, 5500])  # [DL]
        self.width = np.array([5.0, 3.0, 3.0, 4499])  # [DL]
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)  # [DL]

        # IC radiation properties
        self.electron_energy, self.radiation_matrix = self.set_radiation_matrix(take_each=5)

        # synchrotron radiation parameters
        sin2_avg = 2 / 3
        e = cst.e.gauss.value * Franklin
        self.sharp_8uG = ((2 * e ** 4 / (3 * cst.m_e ** 4 * cst.c ** 7) * (8e-6 * Gauss) ** 2 * sin2_avg).to(
            u.eV ** (-1) * u.yr ** (-1)))

        t_min = synchrotron_timescale(energy=1e18 * u.eV, bfield=8e-6 * Gauss).to(u.yr)
        lg_t_min = np.log10(t_min.value)
        arranged_b2times = np.logspace(lg_t_min, 5, 10000) * u.yr  # [G2 yr]
        self.b2times = arranged_b2times  # np.linspace(0, 1e5, 10000) * u.yr

        # transport solutions
        e_b2t, b2t_e = np.meshgrid(self.electron_energy, arranged_b2times, indexing='ij')
        allowed = self.sharp_8uG * b2t_e * e_b2t < 1
        allowed_energy = e_b2t[allowed]

        # energy backpropagation
        self.e_mod = np.ones_like(e_b2t) * 1e35  # [eV], very high energy instead of non-allowed energies
        self.e_mod[allowed] = allowed_energy / (1 - self.sharp_8uG * b2t_e * e_b2t)[allowed]

        # modulation coefficient
        self.phi_matrix = np.zeros(e_b2t.shape)  # [DL], modulation 0 for non-allowed energies
        self.phi_matrix[allowed] = ((1 - self.sharp_8uG * b2t_e * e_b2t)[allowed]) ** (-2)

    def cooled_spectrum(self, parameters):
        eta = parameters[0]
        gamma = parameters[1]
        k2 = parameters[2]
        time_index = int(parameters[-1])

        # backpropagated energies
        b2t1 = self.b2times[:time_index + 1]
        mod_e1 = self.e_mod[:, :time_index + 1]

        # electron number density after cooling
        q_matrix = self.spectrum.dn_de(mod_e1, eta, gamma, self.k10, k2)
        phi_matrix = self.phi_matrix[:, :time_index + 1]
        dN_dE = trapezoid(q_matrix * phi_matrix, b2t1)
        return (dN_dE / self.b2times[time_index]).to(1 / u.eV)


class JointCooledSpectrumFit(CooledSpectrumFit):
    def __init__(self, nsteps=2000, nwalkers=32, bvalue: float = 1e-6):
        super().__init__(nwalkers, nsteps)

        # MCMC parameters
        self.ndim = 3  # two spectral parameters (norm + pwl) and b2t
        # source parameters   eta gamma  B^2t
        self.mean = np.array([0.0, 2.0, 5000])  # [DL]
        self.width = np.array([5.0, 3.0, 4999])  # [DL]
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)  # [DL]

        # electron cooling precomputed
        times, energies, self.modulation_coefficient = (
            pickle.load(open(os.path.join(JOINT_COOLING_DIR, f"joint_cooling_{bvalue * 1e6:.1f}.pck"), "rb")))
        self.bfield = bvalue * Gauss
        self.times = times
        self.energies = energies * u.eV
        electron_energy = self.energies[:, 0]
        self.synch_timescale = synchrotron_timescale(electron_energy, self.bfield)
        lg_electron_energy = np.log10(electron_energy.value)  # initial energy

        self.electron_energy, self.radiation_matrix = self.set_radiation_matrix(electron_energy=electron_energy)

    def cooled_spectrum(self, parameters):
        # extract the parameters
        eta, gamma, time_index = parameters[0], parameters[1], int(parameters[2])

        # get the backpropagated energy
        e_time = np.clip(self.energies[:, time_index].value, a_min=1e-10, a_max=1e45) * u.eV
        e_allowed = self.synch_timescale > self.times[time_index]

        # calculate the spectrum at that energy
        dn_de1 = self.spectrum.dn_de(e_time, eta=eta, p=gamma, k1=self.k10, k2=self.k20) * e_allowed

        # multiply by the corresponding modulation coefficient
        dn_de = dn_de1 * self.modulation_coefficient[:, time_index]
        return dn_de


class JointSteadyStateFit(CooledSpectrumFit):
    def __init__(self, nsteps=2000, nwalkers=32, bvalue: float = 1e-6):
        super().__init__(nwalkers, nsteps)
        self.bfield = bvalue * Gauss

        # source parameters   eta gamma  k2 t_index
        self.mean = np.array([0.0, 2.5, 4.0, 5000])  # [DL]
        self.width = np.array([5.0, 3.0, 4.0, 4999])  # [DL]
        self.ndim = self.mean.size
        self.start = self.mean + self.width * (np.random.random([self.nwalkers, self.ndim]) - 0.5)  # [DL]

        # electron cooling precomputed
        times, energies, modulation_coefficient = (
            pickle.load(open(os.path.join(JOINT_COOLING_DIR, f"joint_cooling_{bvalue * 1e6:.1f}.pck"), "rb")))
        self.times = times
        self.energies = energies[::2] * u.eV
        electron_energy = self.energies[:, 0]
        self.modulation_coefficient = modulation_coefficient[::2]
        self.synch_timescale = synchrotron_timescale(electron_energy, self.bfield)

        # IC radiation properties
        self.electron_energy, self.radiation_matrix = self.set_radiation_matrix(electron_energy=electron_energy)

    def cooled_spectrum(self, parameters):
        # extract the parameters
        eta = parameters[0]
        gamma = parameters[1]
        k2 = parameters[2]
        time_index = int(parameters[-1])

        # get the backpropagated energies at all times t' <= t
        e_time1 = np.clip(self.energies[:, :time_index + 1], a_min=1e-10 * u.eV, a_max=1e35 * u.eV)

        # injection matrix Q_ij
        q_matrix = self.spectrum.dn_de(e_time1, eta=eta, p=gamma, k1=self.k10, k2=k2)
        # modulation matrix Phi_ij
        phi_matrix = self.modulation_coefficient[:, :time_index + 1]

        # multiply by the corresponding modulation coefficient
        dn_de = trapezoid(q_matrix * phi_matrix, self.times[:time_index + 1])
        return (dn_de / self.times[time_index]).to(1 / u.eV)


def electron_synchrotron_only(nsteps=2000, nwalkers=32):
    synch = SynchrotronCooledSpectrumFit(nsteps=nsteps, nwalkers=nwalkers)
    result = synch.run()
    with open(os.path.join(MCMC_ELECTRONS_SYNCH_ONLY,
                           f"synch_{nsteps}n_{nwalkers}w_4_param_chain.pck"), "wb") as file_open:
        pickle.dump([result, synch], file_open)
    return


def electron_joint_single_cooling(nsteps=2000, nwalkers=32, bvalue=1e-6):
    joint = JointCooledSpectrumFit(nsteps=nsteps, nwalkers=nwalkers, bvalue=bvalue)
    joint.k20 = 5.0
    joint.k10 = -3.0
    result = joint.run()
    with open(os.path.join(MCMC_ELECTRONS_JOINT,
                           f"joint_{nsteps}n_{nwalkers}w_{bvalue * 1e6:.1f}uG.pck"), "wb") as file_open:
        pickle.dump([result, joint], file_open)
    return


def electron_joint_steady_cooling(nsteps=2000, nwalkers=32, bvalue: float = 1e-6):
    joint = JointSteadyStateFit(nsteps=nsteps, nwalkers=nwalkers, bvalue=bvalue)
    joint.k20 = 4.0
    joint.k10 = -10.0
    result = joint.run()
    with open(os.path.join(MCMC_ELECTRONS_STEADY_JOINT,
                           f"joint_{nsteps}n_{nwalkers}w_{bvalue * 1e6:.1f}uG.pck"), "wb") as file_open:
        pickle.dump([result, joint], file_open)
    return


def electron_synch_steady_cooling(nsteps=2000, nwalkers=32):
    synch = SynchrotronSteadyStateFit(nsteps=nsteps, nwalkers=nwalkers)
    print("Synch-Steady solver set")
    result = synch.run()
    with open(os.path.join(MCMC_ELECTRONS_SYNCH_STEADY,
                           f"joint_{nsteps}n_{nwalkers}w.pck"), "wb") as file_open:
        pickle.dump([result, synch], file_open)
    return


if __name__ == '__main__':
    n = 600
    nw = 32
    # electron_synchrotron_only(nsteps=n, nwalkers=nw)
    # electron_joint_single_cooling(nsteps=n, nwalkers=nw, bvalue=6e-6)
    # electron_synch_steady_cooling(nsteps=n, nwalkers=nw)
    electron_joint_steady_cooling(nsteps=n, nwalkers=nw, bvalue=5e-6)
