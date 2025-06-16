import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, galactocentric_frame_defaults

galactocentric_frame_defaults.set('v4.0')


def proton_density_Misiriotis(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2)  # cylindrical

    # molecular hydrogen
    rho_H2_00 = 4.06 * u.cm ** (-3)
    h_H2, z_H2 = 2.57 * u.kpc, 0.08 * u.kpc
    rho_H2 = rho_H2_00 * np.exp(-r / h_H2 - np.abs(z) / z_H2)

    # atomic hydrogen
    rho_HI_00 = 0.32 * u.cm ** (-3)
    h_HI, z_HI = 18.24 * u.kpc, 0.52 * u.kpc
    R_t = 2.75 * u.kpc
    rho_HI = rho_HI_00 * np.exp(-r / h_HI - np.abs(z) / z_HI) * np.heaviside(r - R_t, 1
                                                                             )
    return 2 * rho_H2 + rho_HI


if __name__ == '__main__':
    V4641_ICRS = SkyCoord(
        ra=274.84 * u.deg,
        dec=-25.497 * u.deg,
        distance=(0.1692 * u.mas).to(u.kpc, u.parallax()),
        pm_ra_cosdec=-0.779 * (u.mas / u.yr),
        pm_dec=-0.433 * (u.mas / u.yr),
        radial_velocity=100 * (u.km / u.s),
        frame="icrs",
    )
    Sun_ICRS = SkyCoord(
        ra=0.0 * u.deg,
        dec=0.0 * u.deg,
        distance=0.0 * u.kpc,
        frame="icrs",
    )

    V4641 = V4641_ICRS.transform_to(Galactocentric())
    Sun = Sun_ICRS.transform_to(Galactocentric())
    print(f"Density at the V4641: {proton_density_Misiriotis(*V4641.cartesian.xyz):.2g}")
    print(f"Density at the Sun: {proton_density_Misiriotis(*Sun.cartesian.xyz):.2g}")
