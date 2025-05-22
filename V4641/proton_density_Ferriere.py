import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactic, Galactocentric, galactocentric_frame_defaults
from astropy.coordinates.matrix_utilities import rotation_matrix

_ = galactocentric_frame_defaults.set('v4.0')
galactocentric = Galactocentric()


def ferriere_cord_transformation(x, y, z):
    theta = np.deg2rad(48.5)
    alpha = np.deg2rad(13.5)
    beta = np.deg2rad(20)
    s_t, c_t = np.sin(theta), np.cos(theta)
    s_a, c_a = np.sin(alpha), np.cos(alpha)
    s_b, c_b = np.sin(beta), np.cos(beta)
    # making a left-handed coordinate system by changing x to -x everywhere
    X = -x * c_b * c_t - y * (s_a * s_b * c_t - c_a * s_t) + z * (c_a * s_b * c_t + s_a * c_t)
    Y = +x * c_b * s_t + y * (s_a * s_b * c_t + c_a * s_t) + z * (c_a * s_b * c_t - s_a * c_t)
    Z = -x * s_b + y * s_a * c_b + z * c_a * c_b
    return X, Y, Z


def H2_ferriere_density(X, Y, Z):
    # formula (23)
    n0 = 4.8 * u.cm ** (-3)
    X_d = 1.2 * u.kpc
    L_d = 0.438 * u.kpc
    H_d = 0.042 * u.kpc
    return n0 * np.exp(-((np.sqrt(X**2 + (3.1 * Y)**2) - X_d) / L_d)**4) * np.exp(-(Z / H_d) ** 2)


def HI_ferriere_density(X, Y, Z):
    # formula 24
    n0 = 0.34 * u.cm ** (-3)
    X_d = 1.2 * u.kpc
    L_d = 438 * u.pc
    H_d1 = 120 * u.pc
    return n0 * np.exp(-(np.sqrt(X ** 2 + (3.1 * Y) ** 2) - X_d) / L_d) * np.exp(-(Z / H_d1) ** 2)


def H2_CMZ_ferriere_density(X, Y, Z):
    n0 = 150 * u.cm ** (-3)
    X_c = 125 * u.pc
    L_c = 137 * u.pc
    H_c = 18 * u.pc
    return n0 * np.exp(-((np.sqrt(X ** 2 + (2.5 * Y) ** 2) - X_c) / L_c)**4) * np.exp(-(Z / H_c) ** 2)


def HI_CMZ_ferriere_density(X, Y, Z):
    n0 = 150 * u.cm ** (-3)
    X_c = 125 * u.pc
    L_c = 137 * u.pc
    H_c = 54 * u.pc
    return n0 * np.exp(-(np.sqrt(X ** 2 + (2.5 * Y) ** 2) - X_c) / L_c) * np.exp(-(Z / H_c) ** 2)


def WIM_ferriere_density(x, y, z):
    y_3 = -10 * u.pc
    z_3 = -20 * u.pc
    L_3 = 145 * u.pc
    H_3 = 26 * u.pc
    L_2 = 3.7 * u.kpc
    H_2 = 140 * u.pc
    L_1 = 17 * u.kpc
    H_1 = 950 * u.pc
    n0 = 8.0 * u.cm ** (-3)
    r = np.sqrt(x ** 2 + y ** 2)
    f1 = np.exp(-(x ** 2 + (y - y_3) ** 2) / L_3 ** 2) * np.exp(-(z - z_3) ** 2 / H_3 ** 2)
    f2 = 0.009 * np.exp(-((r - L_2) / (L_2 / 2)) ** 2) / np.cosh(z / H_2 * u.rad) ** 2
    f3 = 0.005 * (np.cos(np.pi * r / (2 * L_1) * u.rad) * np.heaviside(L_1 - r, 0.5)) / np.cosh(z / H_1 * u.rad) ** 2
    return n0 * (f1 + f2 + f3)


def HIM_ferriere_density(x, y, z):
    n0 = 0.0034 * u.cm ** (-3)
    r = np.sqrt(x ** 2 + y ** 2)
    return n0 * np.heaviside(6 * u.kpc - r, 0.5) * np.exp(-(np.abs(z) / (2 * u.kpc))) / 1.2


def get_ferriere_density(x, y, z):
    disk_coordinates = ferriere_cord_transformation(x, y, z)
    H2_density = H2_ferriere_density(*disk_coordinates) + H2_CMZ_ferriere_density(*disk_coordinates)
    HI_density = HI_ferriere_density(*disk_coordinates) + HI_CMZ_ferriere_density(*disk_coordinates)
    H_ionized_density = WIM_ferriere_density(x, y, z) + HIM_ferriere_density(x, y, z)
    return 2 * H2_density + HI_density + H_ionized_density


def plot_density_map():
    # V4641
    V4641_gal = SkyCoord(6.8 * u.deg, -4.8 * u.deg, 6.2 * u.kpc, frame=Galactic)
    V4641 = V4641_gal.transform_to(galactocentric).cartesian

    print(get_ferriere_density(*V4641.xyz))
    # print(get_ferriere_density(0, 0, 0))

    x = np.linspace(-3, 3, 100) * u.kpc
    y = np.linspace(-3, 3, 100) * u.kpc
    z = np.linspace(-3, 3, 100) * u.kpc

    plt.figure(figsize=(10, 5))
    v_min, v_max = 0, 10

    ax1 = plt.subplot(1, 3, 1)
    plt.title("Oxy")
    xy, yx = np.meshgrid(x, y, indexing='ij')
    density = np.zeros_like(xy.value) * u.cm**(-3)
    for z_i in z:
        density += get_ferriere_density(xy, yx, z_i)
    plt.pcolormesh(yx.value, xy.value, density.value, vmin=v_min, vmax=v_max)
    plt.xlabel("y, kpc")
    plt.ylabel("x, kpc")
    plt.scatter(V4641.y.value, V4641.x.value, s=30, c='cyan', marker='*')
    plt.gca().invert_xaxis()

    ax2 = plt.subplot(1, 3, 2)
    plt.title("Oyz")
    yz, zy = np.meshgrid(y, z, indexing='ij')
    density = np.zeros_like(yz.value) * u.cm ** (-3)
    for x_i in x:
        density += get_ferriere_density(x_i, yz, zy)
    plt.pcolormesh(yz.value, zy.value, density.value, vmin=v_min, vmax=v_max)
    plt.scatter(V4641.y.value, V4641.z.value, s=30, c='cyan', marker='*')
    plt.xlabel("y, kpc")
    plt.ylabel("z, kpc")
    plt.gca().invert_xaxis()

    ax3 = plt.subplot(1, 3, 3)
    plt.title("Oxz")
    xz, zx = np.meshgrid(x, z, indexing='ij')
    yy = np.zeros_like(xz)
    density = np.zeros_like(yz.value) * u.cm ** (-3)
    for y_i in y:
        density += get_ferriere_density(xz, y_i, zx)
    pcm = plt.pcolormesh(xz.value, zx.value, density.value, vmin=v_min, vmax=v_max)
    plt.scatter(V4641.x.value, V4641.z.value, s=30, c='cyan', marker='*')
    plt.xlabel("x, kpc")
    plt.ylabel("z, kpc")

    plt.tight_layout()

    plt.colorbar(pcm, ax=[ax1, ax2, ax3], shrink=0.3, location='bottom')
    plt.show()

    return


if __name__ == '__main__':
    plot_density_map()
