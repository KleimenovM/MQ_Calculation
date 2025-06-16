import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactic, ICRS

from astropy.coordinates import Galactocentric, galactocentric_frame_defaults
from mpmath.libmp import normalize


def length(v):
    return np.sqrt(np.dot(v, v))


def unit_vector(v):
    return v / length(v)


def angle(v1, v2):
    return np.rad2deg(np.arccos(np.dot(v1, v2) / (length(v1) * length(v2))))


def projection(v1, v2):
    """
    Project v1 onto v2.

    Parameters:
    v1 : The vector to be projected.
    v2 : The vector onto which v1 is being projected.

    Returns: The projection of v1 onto v2.
    """
    return np.dot(v1, v2) / length(v2) ** 2 * v2


def two_point_line(p1, p2, axis, **kwargs):
    axis.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], **kwargs)
    return


def axes(ax):
    # Set an equal aspect ratio
    lim_value = 8
    ax.set_xlim(-lim_value, lim_value)
    ax.set_ylim(-lim_value, lim_value)
    ax.set_zlim(-lim_value, lim_value)
    ax.set_aspect('equal')

    # axes
    ax_value = 9
    axis_color = 'grey'
    axis_alpha = 0.7
    plt.plot((-ax_value, ax_value), (0, 0), (0, 0), color=axis_color, alpha=axis_alpha)
    plt.plot((0, 0), (-ax_value, ax_value), (0, 0), color=axis_color, alpha=axis_alpha)
    plt.plot((0, 0), (0, 0), (-ax_value, ax_value), color=axis_color, alpha=axis_alpha)

    # titles
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return


def rotate_the_bar(x1, y1, z1, angle_1, angle_2=0.0 * u.deg):
    coord_matrix = np.array([x1, y1, z1])

    rotation_matrix_xz = rotation_matrix(angle_2, 'y')
    rotation_matrix_xy = rotation_matrix(angle_1, 'z')

    full_rm = rotation_matrix_xz @ rotation_matrix_xy
    bar_rot = np.einsum('ij,jkl', full_rm, coord_matrix)
    return bar_rot, full_rm


def galactic_bulge(ax, angle1, angle2=0 * u.deg):
    a = 2.2 * u.kpc
    b = 1.4 * u.kpc
    c = 1.2 * u.kpc
    n = 51
    w = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x1 = a * np.outer(np.cos(w), np.sin(v))
    y1 = b * np.outer(np.sin(w), np.sin(v))
    z1 = c * np.outer(np.ones(np.size(w)), np.cos(v))

    bar_rot, full_rm = rotate_the_bar(x1, y1, z1, angle1, angle2)

    def is_in_the_bar(vec, name="V4146"):
        x_s_bar, y_s_bar, z_s_bar = full_rm.T @ vec
        print(f"Dist from the center: {length(vec):.3f}, coords: {vec}")
        if_in = (x_s_bar / a) ** 2 + (y_s_bar / b) ** 2 + (z_s_bar / c) ** 2 < 1
        print(f"{name} is{'' if if_in else ' not'} in the bulge")
        return

    a_axis_vector = bar_rot[:, n // 2, n // 2]
    b_axis_vector = bar_rot[:, n // 4, n // 2]
    c_axis_vector = bar_rot[:, 0, 0]
    axis_vectors = [a_axis_vector, b_axis_vector, c_axis_vector]

    # plot the bulge
    bar_color = 'orange'
    bar_alpha = 0.1
    bar_axes_alpha = 1.0
    ax.plot_surface(bar_rot[0], bar_rot[1], bar_rot[2], alpha=bar_alpha, color=bar_color)
    for v in axis_vectors:
        plot_v = 1.1 * v
        two_point_line(plot_v, -plot_v, ax, color=bar_color, alpha=bar_axes_alpha)

    return axis_vectors, is_in_the_bar


def galactic_bar(ax, angle1, angle2=0 * u.deg):
    scale = 4.6 * u.kpc
    a = 1 * scale
    b = 1/4 * scale
    c = 1/12 * scale
    n = 51
    w = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x1 = a * np.outer(np.cos(w), np.sin(v))
    y1 = b * np.outer(np.sin(w), np.sin(v))
    z1 = c * np.outer(np.ones(np.size(w)), np.cos(v))

    bar_rot, full_rm = rotate_the_bar(x1, y1, z1, angle1, angle2)

    def is_in_the_bar(vec, name="V4146"):
        x_s_bar, y_s_bar, z_s_bar = full_rm.T @ vec
        print(f"Dist from the center: {length(vec):.3f}, coords: {vec}")
        if_in = (x_s_bar / a) ** 2 + (y_s_bar / b) ** 2 + (z_s_bar / c) ** 2 < 1
        print(f"{name} is{'' if if_in else ' not'} in the bar")
        return

    a_axis_vector = bar_rot[:, n // 2, n // 2]
    b_axis_vector = bar_rot[:, n // 4, n // 2]
    c_axis_vector = bar_rot[:, 0, 0]
    axis_vectors = [a_axis_vector, b_axis_vector, c_axis_vector]

    # plot the bar
    bar_color = 'orangered'
    bar_alpha = 0.1
    bar_axes_alpha = 1.0
    ax.plot_surface(bar_rot[0], bar_rot[1], bar_rot[2], alpha=bar_alpha, color=bar_color)
    for v in axis_vectors:
        plot_v = 1.1 * v
        two_point_line(plot_v, -plot_v, ax, color=bar_color, alpha=bar_axes_alpha)

    return axis_vectors, is_in_the_bar


def galactic_center(ax, galactocentric):
    # Galactic center
    gc = SkyCoord(0, 0, 0, frame=galactocentric).cartesian
    ax.scatter(*gc.xyz, marker='o', color='black', label='Galactic center')
    return gc


def sun(ax, galactocentric):
    Sun_gal = SkyCoord(0 * u.deg, 0 * u.deg, 0 * u.kpc, frame=Galactic)
    Sun = Sun_gal.transform_to(galactocentric).cartesian
    ax.scatter(*Sun.xyz, marker='o', color='red', label='Sun')

    # Solar orbit
    R_disk = length(Sun.xyz)  # kpc
    phi = np.linspace(0, 2 * np.pi, 100)
    x = R_disk * np.cos(phi)
    y = R_disk * np.sin(phi)
    z = np.zeros_like(x)
    ax.plot(x, y, z, 'gray', alpha=0.7)
    return Sun


def sun_velocity(ax, Sun, sun_velocity_value):
    sun_velocity_vector = sun_velocity_value * unit_vector(np.cross(Sun.xyz, [0, 0, 1]))
    print(f"Velocity of the source in the bar: {length(sun_velocity_vector):.0f} km/s")
    return sun_velocity_vector


def v4641(ax, galactocentric, dist):
    V4641_icrs = SkyCoord(18 * u.h + 19.36 * u.min, -25 * u.deg - 24.26 * u.arcmin,
                          distance=dist, frame=ICRS)
    V4641 = V4641_icrs.transform_to(galactocentric).cartesian
    ax.scatter(*V4641.xyz, marker='s', color='purple', label='V4641', s=40)
    return V4641


def v4641_velocity(ax, V4641, Omega_bar, bar_axes):
    # velocity of the source in the bar
    velocity_value = Omega_bar * length(V4641.xyz)  # [km/s]
    V4641_hor = np.cross(V4641.xyz, bar_axes[2])
    velocity_vector = velocity_value * unit_vector(V4641_hor)
    print(f"Velocity of the source in the bar: {length(velocity_vector):.0f} km/s")
    return velocity_vector


def line_of_sight_vector(ax, source, Sun):
    los_vector = source.xyz - Sun.xyz
    los_unit_vector = unit_vector(los_vector)
    # two_point_line(Sun.xyz.value, source.xyz.value, ax, color='green', label='LOS')
    return los_unit_vector


def low_energy_jet(ax, galactocentric, los_unit_vector, V4641, dist):
    V_4641_jet_icrs = SkyCoord(18 * u.h + 19 * u.min + 21.646 * u.s,
                               -(25 * u.deg + 24 * u.arcmin + 25.65 * u.arcsec),
                               distance=dist, frame=ICRS)
    V_4641_jet_projected = V_4641_jet_icrs.transform_to(galactocentric).cartesian
    jet_los_angle = np.deg2rad(-15)  # angle between jet and line of sight in degrees [SRC-2020]
    jet_1 = V_4641_jet_projected.xyz - V4641.xyz
    jet_2 = - los_unit_vector * length(jet_1) / np.tan(jet_los_angle)
    jet = jet_1 + jet_2
    vec_len = .1 / np.cos(np.deg2rad(75))
    # ax.quiver(*V4641.xyz, *jet, length=vec_len, normalize=True, color='red', label='radio jet')
    # ax.quiver(*V4641.xyz, *(-jet), length=vec_len, normalize=True, color='red')
    return jet


def high_energy_jet(ax, galactocentric, V4641, dist, size_north=0.1 * u.kpc, size_south=0.1 * u.kpc):
    V4641_North_icrs = SkyCoord(18 * u.h + 19.44 * u.min,
                                -25 * u.deg - 0 * u.arcmin,
                                distance=dist, frame=ICRS)
    V4641_South_icrs = SkyCoord(18 * u.h + 19.56 * u.min,
                                -26 * u.deg - 5 * u.arcmin,
                                distance=dist, frame=ICRS)
    V4641_North = V4641_North_icrs.transform_to(galactocentric).cartesian
    V4641_South = V4641_South_icrs.transform_to(galactocentric).cartesian

    UHE_vec_North = unit_vector((V4641_North - V4641).xyz) * size_north
    UHE_vec_South = unit_vector((V4641_South - V4641).xyz) * size_south

    ax.quiver(*V4641.xyz, *UHE_vec_North, color='blue', label='high-energy jet')
    ax.quiver(*V4641.xyz, *UHE_vec_South, color='blue')
    return V4641.xyz + UHE_vec_North, V4641.xyz + UHE_vec_South


def project_to_bar_axis(ax, uhe_north, uhe_south, V4641, Sun, bar_axes):
    n = bar_axes[1]  # b-axis of the bar
    v_north_jet = uhe_north - Sun.xyz
    v_south_jet = uhe_south - Sun.xyz
    v_source = (V4641 - Sun).xyz

    t_north = np.dot(v_source, n) / np.dot(v_north_jet, n)
    R_north = v_north_jet * t_north - v_source

    t_south = np.dot(v_source, n) / np.dot(v_south_jet, n)
    R_south = v_south_jet * t_south - v_source

    ax.quiver(*V4641.xyz, *R_north, color='green', label='projected jet')
    ax.quiver(*V4641.xyz, *R_south, color='green')

    print(f"angle {angle(R_north, uhe_north - V4641.xyz):.1f}, length {length(R_north):.3f}")
    print(f"angle {angle(R_south, uhe_south - V4641.xyz):.1f}, length {length(R_south):.3f}")

    return V4641.xyz + R_north, V4641.xyz + R_south


def plot_the_source_and_the_bar():
    _ = galactocentric_frame_defaults.set('v4.0')
    galactocentric = Galactocentric()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    axes(ax)  # equalize and draw

    # galactic center
    GC = galactic_center(ax, galactocentric)

    # Galactic bulge & bar
    bar_angle = 27 * u.deg
    bulge_vecs, is_in_the_bulge = galactic_bulge(ax, bar_angle)
    bar_vecs, is_in_the_bar = galactic_bar(ax, bar_angle)

    # Sun
    Sun = sun(ax, galactocentric)
    # Sun velocity
    Sun_velocity = sun_velocity(ax, Sun, 220 * u.km/u.s)

    # V4641 point source
    dist = 6.1 * u.kpc
    omega_bar = 120 * u.km/u.s/u.kpc
    V4641 = v4641(ax, galactocentric, dist)

    # line of sight vector
    los_unit_vector = line_of_sight_vector(ax, V4641, Sun)

    # V4641 absolute and relative velocity and its projection onto the line of sight
    V4641_vel_abs = v4641_velocity(ax, V4641, omega_bar, bar_vecs)
    V4641_vel_rel = V4641_vel_abs - Sun_velocity
    V4641_radial = projection(V4641_vel_rel, los_unit_vector)
    print(f"Radial velocity as seen from Earth: {length(V4641_radial):.0f}")

    # Solar velocity

    # V4641 low-energy jet
    jet = low_energy_jet(ax, galactocentric, los_unit_vector, V4641, dist)

    # V4641 high-energy jet [H.E.S.S. preliminary, 2024]
    UHE_north, UHE_south = high_energy_jet(ax, galactocentric, V4641, dist)
    R_north, R_south = project_to_bar_axis(ax, UHE_north, UHE_south, V4641, Sun, bulge_vecs)

    dist_r_north = np.sqrt(R_north[0] ** 2 + R_north[1] ** 2)
    dist_r_south = np.sqrt(R_south[0] ** 2 + R_south[1] ** 2)
    print(f"{dist_r_north:.3f}, {dist_r_south:.3f}")

    ax.view_init(elev=90, azim=180, roll=0)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    plot_the_source_and_the_bar()
