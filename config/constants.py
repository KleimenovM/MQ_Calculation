import astropy.units as u
from astropy.constants import codata2018 as cst
from scipy.special import zeta

from config.units import Franklin

CST_HC = cst.h * cst.c

CST_e = cst.e.gauss.value * Franklin

CST_m_e = (cst.m_e * cst.c**2).to(u.eV)

T_CMB = 2.72548 * u.K  # [K], CMB temperature, Source: # https://en.wikipedia.org/wiki/Cosmic_microwave_background
E_CMB = 3 * zeta(4) / zeta(3) * T_CMB  # [eV]

