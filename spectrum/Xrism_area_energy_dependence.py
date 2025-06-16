# a module to calculate the XRISM measurement area at different energies
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import astropy.units as u

from analysis_electrons.general_spectrum_analysis import volume_approximation, width_approximation, length_approximation
from config.plotting import set_plotting_defaults


class XrismArea:
    def __init__(self, dist = 6.1 * u.kpc):
        self.dist = dist
        self.arcmin_to_pc = self.dist.to(u.pc) * np.deg2rad(1/60) / u.arcmin
        self.r1 = 12 * u.arcmin * self.arcmin_to_pc
        self.r2 = 24 * u.arcmin * self.arcmin_to_pc

        self.center = -45 * u.deg
        self.half_width = 23.5 * u.deg

        self.cmin, self.cmax = np.cos(self.center - self.half_width), np.cos(self.center + self.half_width)
        self.smin, self.smax = np.sin(self.center - self.half_width), np.sin(self.center + self.half_width)

    def _which_in(self, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        cos_phi = x / r
        sin_phi = y / r

        inner_circle = (r <= self.r1)
        outer_circle = (self.r1 < r) * (r <= self.r2)

        outer_slice = (cos_phi >= self.cmin) & (cos_phi <= self.cmax) & (sin_phi >= self.smin) & (sin_phi <= self.smax)

        return inner_circle + (outer_circle * outer_slice)

    def get_contours(self):
        phi1 = np.linspace(self.center + self.half_width, 360 * u.deg + self.center - self.half_width, 100)
        x1, y1 = self.r1 * np.cos(phi1), self.r1 * np.sin(phi1)

        phi21, phi22 = self.center - self.half_width, self.center + self.half_width
        r2 = np.linspace(self.r1, self.r2, 50)
        x21, y21 = r2 * np.cos(phi21), r2 * np.sin(phi21)
        x22, y22 = r2 * np.cos(phi22), r2 * np.sin(phi22)

        phi3 = np.linspace(self.center - self.half_width, self.center + self.half_width, 50)
        x3, y3 = self.r2 * np.cos(phi3), self.r2 * np.sin(phi3)

        return np.concatenate([x3, x22, x1, x21]), np.concatenate([y3, y22, y1, y21])

    def get(self, x, y):
        answer = self._which_in(x, y)
        return x[answer], y[answer]

    def count(self, x, y):
        answer = self._which_in(x, y)
        return np.sum(answer)


def xrism_in_out_ratio(electron_energy, dist=6.1 * u.kpc,
                       lw_ratio: float = 3.56,
                       size_HESS_deg=60*u.arcmin,
                       energy_HESS=10*u.TeV):
    amin_pc = (1/180*np.pi * dist).to(u.pc) / (60 * u.arcmin)
    XRISM_area = (629.4 * u.arcmin ** 2) * amin_pc ** 2

    wl_ratio = 1 / lw_ratio
    size_HESS = size_HESS_deg * amin_pc

    size_1TeV = size_HESS * (1 * u.TeV / energy_HESS) ** (1 / 6)
    print(size_1TeV)

    e_width = width_approximation(electron_energy, width_length_ratio=wl_ratio, size_1TeV=size_1TeV)
    e_length = length_approximation(electron_energy, size_1TeV=size_1TeV)

    n_dots = 10000

    uniform_slab = stats.uniform(loc=[-1.0, -1.0], scale=[2.0, 2.0])
    diffusive_slab = stats.norm(loc=[0.0, 0.0], scale=[0.5, 0.5])

    p_u = uniform_slab.rvs(size=[n_dots, 2])
    x_u, y_u = p_u[:, 0], p_u[:, 1]

    p_d = diffusive_slab.rvs(size=[n_dots, 2])
    x_d, y_d = p_d[:, 0], p_d[:, 1]

    xa = XrismArea()

    def XRISM_ratio_uniform(hw_x, hw_y):
        x1_u, y1_u = x_u * hw_x, y_u * hw_y
        return xa.count(x1_u, y1_u) / n_dots

    def XRISM_ratio_diffusive(hw_x, hw_y):
        x1_d, y1_d = x_d * hw_x, y_d * hw_y
        return xa.count(x1_d, y1_d) / n_dots

    norm_dA, dif_dA = [], []
    for i, e in enumerate(electron_energy):
        hw_x1, hw_y1 = e_width[i] / 2, e_length[i] / 2
        norm_dA.append(XRISM_ratio_uniform(hw_x1, hw_y1))
        dif_dA.append(XRISM_ratio_diffusive(hw_x1, hw_y1))

    norm_dA, dif_dA = np.array(norm_dA), np.array(dif_dA)
    return norm_dA, dif_dA


if __name__ == '__main__':
    print("Not for direct use")
