import os.path
import pickle
from pathlib import Path

import numpy as np

from scipy.interpolate import interp1d

from astropy.constants import codata2010 as const
from astropy.coordinates import SkyCoord, Galactocentric, galactocentric_frame_defaults
import astropy.units as u

galactocentric_frame_defaults.set('v4.0')
galactocentric = Galactocentric()

t_cmb = 2.72548 * u.K  # [K], CMB temperature, Source: # https://en.wikipedia.org/wiki/Cosmic_microwave_background

ROOT_DIR = Path(__file__).resolve().parent.parent
WEIN_DATA_FOLDER = os.path.join(ROOT_DIR, 'GalacticExtinction')


def wvl_to_e(wvl):
    """
    Energy of a photon with wavelenth wvl (e = hc / wvl)
    :param wvl: [mkm], wavelength
    :return: [eV], energy
    """
    return (const.h * const.c / wvl).to(u.eV)


def e_to_wvl(e):
    """
    Wavelength of a photon with energy e (wvl = hc / e)
    :param e: [eV], energy
    :return: [mkm], wavelength
    """
    return (const.h * const.c / e).to(u.um)


class BBR:
    def __init__(self, t):
        self.t = t
        return

    def intensity(self, wvl, z):
        # wavelength redshift
        wvl_z = wvl * (1 + z)
        # temperature redshift
        t = self.t * (1 + z)
        theta = (const.h * const.c / (wvl_z * const.k_B * t)).clip(1e-10, 1e2)
        lower_exp_part = 1 / (np.exp(theta) - 1)
        return 2 * u.sr ** (-1) * const.h * const.c ** 2 / wvl_z ** 4 * lower_exp_part


class CosmicBackground:
    def __init__(self, cmb_on: bool = False):
        self.cmb = BBR(t_cmb)
        self.cmb_on = cmb_on

    def no_cmb_intensity(self, wvl, z):
        """
        Get the peculiar EBL intensity by wavelength (per wavelength unit)
        :param z: [DL], redshift
        :param wvl: [mkm], photon wavelength
        :return: [W m-2 sr-1], EBL intensity
        """
        return .0 * (u.W * u.m ** (-2) * u.sr ** (-1))

    def intensity(self, wvl, z):
        """
        Get the EBL intensity by wavelength (per wavelength unit)
        :param z: [DL], redshift
        :param wvl: [mkm], photon wavelength
        :return: [W cm-2 sr-1], EBL intensity
        """
        intensity = self.no_cmb_intensity(wvl, z) + self.cmb.intensity(wvl, z) * self.cmb_on
        return intensity.to(u.W * u.cm ** (-2) * u.sr ** (-1))

    def density_e(self, e, z):
        """
        Get the EBL spectral number density [m-3 eV-1]
        :param e: [eV], photon energy
        :param z: [DL], redshift
        :return: [cm-3 eV-1], spectral number density
        """
        density = 4 * np.pi * u.sr / const.c * self.intensity(e_to_wvl(e), z) * e ** (-2) * (1 + z) ** 3
        return density.to(u.eV ** (-1) * u.cm ** (-3))


class CMBOnly(CosmicBackground):
    def __init__(self):
        super().__init__(True)


class Dust:
    def __init__(self, if_new=True):
        self.if_new = if_new
        self.rho_0c = 1.51e-25 * u.g * u.cm ** (-3)  # cold density
        self.rho_0w = 1.22e-27 * u.g * u.cm ** (-3)  # warm density
        self.R_c = 5 * u.kpc
        self.R_w = 3.3 * u.kpc
        self.Z_c = 0.1 * u.kpc
        self.Z_w = 0.09 * u.kpc
        self.T_0c = 19.2 * u.K
        self.R_T = 48 * u.kpc
        self.Z_T = 500 * u.kpc
        self.T_w = 35.0 * u.K
        self.T_inf = 2.7255 * u.K
        self.Weingarten_paths = ["Weingarten_2001_lambda.txt",
                                 "Weingarten_2003_lambda.txt"]
        self.k_wvl_interp = self.__get_k_wvl()

    def rho_c(self, R, Z):
        return self.rho_0c * np.exp(-R / self.R_c - np.abs(Z) / self.Z_c)

    def rho_w(self, R, Z):
        return self.rho_0w * np.exp(-R / self.R_w - np.abs(Z) / self.Z_w)

    def t_c(self, R, Z):
        return (self.T_0c - self.T_inf) * np.exp(-R / self.R_T - np.abs(Z) / self.Z_T) + self.T_inf

    def __get_k_wvl(self):
        """
        ! internal function, not for direct use !
        Read Weingarten et al. emissivity from a file
        Source: http://www.astro.princeton.edu/~draine/dust/dustmix.html, (R_V = 3.1, 2003)
        :return: wavelength-emissivity interpolator
        """
        fn = os.path.join(WEIN_DATA_FOLDER, self.Weingarten_paths[int(self.if_new)])
        with open(fn) as f:
            d = f.readlines()

        wvl, res = [], []
        for line in d[4:]:
            vals = line.split()
            wvl.append(float(vals[0].strip()))
            res.append(float(vals[4].strip()))

        return interp1d(wvl, res)

    def k_wvl(self, wvl):
        """
        Get the Weingarten emissivity
        :param wvl: [mkm], photon wavelength
        :return: emissivity [cm**2 / g]
        """
        return self.k_wvl_interp(wvl.value) * u.cm ** 2 / u.g

    def get_eta(self, wvl, R, Z, z=0):
        """
        Get the power emitted per unit volume per solid angle per volume per wvl
        :param wvl: [mkm], photon wavelength
        :param R: [kpc], radial galactic coordinate
        :param Z: [kpc], vertical galactic coordinate
        :param z: [DL], redshift (taken 0, used just for compatibility)
        :return: eta_lambda, [W cm-3 sr-1 mkm-1]
        """
        # cold component, lambda * eta_lambda
        wvl_eta_c = self.rho_c(R, Z) * self.k_wvl(wvl) * BBR(self.t_c(R, Z)).intensity(wvl, z)
        # warm component, lambda * eta_lambda
        wvl_eta_w = self.rho_w(R, Z) * self.k_wvl(wvl) * BBR(self.T_w).intensity(wvl, z)
        return ((wvl_eta_w + wvl_eta_c) / wvl).to(u.W * u.cm ** (-3) * u.sr ** (-1) * u.um ** (-1))


class Stars:
    def __init__(self):
        # from [Vernetto-2016]
        starlight_data = np.loadtxt("starlight_density_at_the_Sun.txt", skiprows=1, unpack=True, delimiter=",")
        wvl = starlight_data[0] * u.um
        self.e = wvl_to_e(wvl)

        l_ul = starlight_data[1] * u.eV * u.cm**(-3)
        self.energy_density_e = (l_ul / self.e).to(u.cm**(-3))

        self.R = 2.17 * u.kpc
        self.Z = 7.22 * u.kpc

        # Sun in GC frame
        sun = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, distance=0.0 * u.kpc, frame='icrs').transform_to(galactocentric)
        sun_cords = sun.cartesian.xyz
        r_sun, z_sun = np.sqrt(sun_cords[0]**2 + sun_cords[1]**2), np.abs(sun_cords[2])

        # starlight photon density at the GC
        self.energy_density_GC = self.energy_density_e * np.exp(+r_sun/self.R + z_sun/self.Z)
        self.interpolator = self.interpolate_energy_density()

    def interpolate_energy_density(self):
        lg_e = np.log10(self.e / u.eV)
        lg_u_e = np.log10(self.energy_density_GC / (u.cm ** (-3)))
        interpolator = interp1d(lg_e[::-1], lg_u_e[::-1], bounds_error=False, fill_value='extrapolate')
        return interpolator

    def u_e(self, e, r, z):
        lg_e = np.log10(e / u.eV)
        return 10**self.interpolator(lg_e) * u.cm**(-3) * np.exp(-r / self.R - np.abs(z) / self.Z)

    def density_e(self, e, r, z):
        return self.u_e(e, r, z) / e


if __name__ == '__main__':
    print("Not for direct use")
