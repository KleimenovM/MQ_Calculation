import astropy.units as u
from astropy.constants import codata2018 as cst
from scipy.special import zeta

CST_HC = cst.h * cst.c

T_CMB = 2.72548 * u.K  # [K], CMB temperature, Source: # https://en.wikipedia.org/wiki/Cosmic_microwave_background
E_CMB = 3 * zeta(4) / zeta(3) * T_CMB  # [eV]

