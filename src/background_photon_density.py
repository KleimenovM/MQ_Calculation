import os.path
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d, RegularGridInterpolator, interpn
from scipy.io import readsav

from astropy.constants import codata2010 as cst
from astropy.coordinates import SkyCoord, Galactocentric, galactocentric_frame_defaults
import astropy.units as u

from config.constants import CST_HC
from config.settings import ISRF_DIR

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
    return (CST_HC / wvl).to(u.eV)


def e_to_wvl(e):
    """
    Wavelength of a photon with energy e (wvl = hc / e)
    :param e: [eV], energy
    :return: [mkm], wavelength
    """
    return (CST_HC / e).to(u.um)


class BBR:
    def __init__(self, t):
        self.t = t
        return

    def intensity(self, wvl, z):
        # wavelength redshift
        wvl_z = wvl * (1 + z)
        # temperature redshift
        t = self.t * (1 + z)
        theta = (CST_HC / (wvl_z * cst.k_B * t)).clip(1e-10, 1e3)
        lower_exp_part = 1 / (np.exp(theta) - 1)
        return 2 * u.sr ** (-1) * cst.h * cst.c ** 2 / wvl_z ** 4 * lower_exp_part


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

    def density_e(self, e, z=0.0):
        """
        Get the EBL spectral number density [m-3 eV-1]
        :param e: [eV], photon energy
        :param z: [DL], redshift
        :return: [cm-3 eV-1], spectral number density
        """
        density = 4 * np.pi * u.sr / cst.c * self.intensity(e_to_wvl(e), z) * e ** (-2) * (1 + z) ** 3
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


class DustLines:
    # [Popescu et al., 2017]
    def __init__(self):
        data_path = os.path.join(ISRF_DIR, "grid_dust.save")
        self.data = readsav(data_path)
        self.r = self.data.rr * u.pc
        self.z = self.data.zz * u.pc
        self.wvl = self.data.lambda_arr * u.um
        rf_unit = u.uJ / u.pc ** 3 / u.nm
        self.rf_unit = u.eV / u.cm ** 3 / u.um
        self.rad_field = (self.data.urad_out_arr * rf_unit).to(self.rf_unit)

    def density_e(self, e, r, z):
        wvl = e_to_wvl(e)
        ul = 10 ** interpn((self.wvl.value, self.z.value, self.r.value),
                           np.log10(self.rad_field.value),
                           (wvl.value, z.value, r.value),
                           bounds_error=False, method='linear',
                           fill_value=None
                           ) * self.rf_unit
        return (wvl * ul / e ** 2).to(1 / (u.eV * u.cm ** 3))


class Starlight:
    # [Popescu et al., 2017]
    def __init__(self):
        data_path = os.path.join(ISRF_DIR, "popescu_radiation.txt")
        self.data = pd.read_csv(data_path, sep='\t')

        dict_path = os.path.join(ISRF_DIR, "popescu_dictionary.txt")
        self.dict = pd.read_csv(dict_path, sep='\t')

        self.wvls, self.r, self.z, self.table = self.set_the_map()
        self.energies = wvl_to_e(self.wvls)

        self.table_e = self.reshape_for_extrapolation()

    def set_the_map(self):
        bands = self.dict['band']
        wvls = self.dict['nm']  # [nm]

        result = []
        r, z = [], []
        for i, b in enumerate(bands):
            data_i = self.data[self.data['band'] == b]
            result_i = []
            r_i = []
            for j, r_j in enumerate(data_i['r(pc)']):
                if r_i.count(r_j) > 0:
                    continue
                r_i.append(r_j)
                data_ij = data_i[data_i['r(pc)'] == r_j]
                if i == 0 and j == 0:
                    z = data_ij['z(pc)']
                density_ij = wvls[i] * data_ij['urad(uJ/pc3/nm)']  # eu_e [uJ/pc3]
                result_i.append(density_ij)
            if i == 0:
                r = r_i
            result.append(result_i)

        return (np.array(wvls) * u.nm,
                np.array(r) * u.pc, np.array(z) * u.pc,
                np.array(result) * u.uJ / u.pc ** 3)

    def reshape_for_extrapolation(self):
        answer = np.zeros([self.energies.size, self.r.size, self.z.size])
        for i in range(self.r.size):
            for j in range(self.z.size):
                answer[:, i, j] = (self.table[:, i, j] / self.energies ** 3).value
        return (np.array(answer) * (u.uJ / u.pc ** 3 / u.eV ** 3)).to(1 / u.eV ** 2 / u.cm ** 3)

    def density_e(self, e, r, z):

        ul = 10 ** interpn((self.energies, self.r, self.z),
                           np.log10(self.table_e.value),
                           (e.value, r.value, z.value),
                           bounds_error=False, method='linear',
                           fill_value=None
                           ) * (1 / u.eV ** 2 / u.cm ** 3) * e ** 3
        return ul / e ** 2


class Stars:
    def __init__(self):
        # from [Vernetto-2016]
        starlight_data = np.loadtxt("starlight_density_at_the_Sun.txt", skiprows=1, unpack=True, delimiter=",")
        wvl = starlight_data[0] * u.um
        self.e = wvl_to_e(wvl)

        l_ul = starlight_data[1] * u.eV * u.cm ** (-3)
        self.energy_density_e = (l_ul / self.e).to(u.cm ** (-3))

        self.R = 2.17 * u.kpc
        self.Z = 7.22 * u.kpc

        # Sun in GC frame
        sun = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, distance=0.0 * u.kpc, frame='icrs').transform_to(galactocentric)
        sun_cords = sun.cartesian.xyz
        r_sun, z_sun = np.sqrt(sun_cords[0] ** 2 + sun_cords[1] ** 2), np.abs(sun_cords[2])

        # starlight photon density at the GC
        self.energy_density_GC = self.energy_density_e * np.exp(+r_sun / self.R + z_sun / self.Z)
        self.interpolator = self.interpolate_energy_density()

    def interpolate_energy_density(self):
        lg_e = np.log10(self.e / u.eV)
        lg_u_e = np.log10(self.energy_density_GC / (u.cm ** (-3)))
        interpolator = interp1d(lg_e[::-1], lg_u_e[::-1], bounds_error=False, fill_value='extrapolate')
        return interpolator

    def u_e(self, e, r, z):
        lg_e = np.log10(e / u.eV)
        return 10 ** self.interpolator(lg_e) * u.cm ** (-3) * np.exp(-r / self.R - np.abs(z) / self.Z)

    def density_e(self, e, r, z):
        return self.u_e(e, r, z) / e


if __name__ == '__main__':
    print("Not for direct use")
