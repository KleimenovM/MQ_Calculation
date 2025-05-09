import astropy.units as u
from astropy.constants import codata2018 as cst
from fractions import Fraction

Franklin = u.g ** Fraction(1, 2) * u.cm ** Fraction(3, 2) * u.s ** -1
Gauss = u.g ** Fraction(1, 2) * u.cm ** (-Fraction(1, 2)) * u.s ** -1

flux_unit = u.erg / (u.cm ** 2 * u.s)
rad_density_unit = u.erg / (u.cm ** 3 * u.s)
rad_unit = u.erg / u.s
