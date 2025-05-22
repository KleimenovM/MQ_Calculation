import numpy as np
import matplotlib.pyplot as plt

from config.plotting import set_plotting_defaults, save_figure


def plot_distance_evolution():
    set_plotting_defaults()
    plt.figure(figsize=(5.5, 3))

    vals = []
    errs = []

    # [Hjellming et al., 1999]
    dist, d_l, d_h = 0.4, 0.0, 0.4
    plt.errorbar(dist, 0, xerr=d_h, xlolims=True, marker='o',
                 linewidth=2, capthick=2)

    # [Orosz et al., 2001]
    dist_min, dist_max = 7.4, 12.31
    dist = np.array([(dist_min + dist_max) / 2])
    d_l, d_h = dist - dist_min, dist_max - dist
    # vals.append(dist)
    # errs.append(d_l)
    plt.errorbar(dist, 1, xerr=[d_l, d_h], marker='o', capsize=4, linewidth=2, capthick=2)

    # [Chaty et al., 2003]
    dist, d_l, d_h = np.array([5.5]), np.array([2.5]), np.array([2.5])
    vals.append(dist)
    errs.append((d_l + d_h) / 2)
    plt.errorbar(dist, 2, xerr=[d_l, d_h], marker='o', capsize=4, linewidth=2, capthick=2)

    # [MacDonald et al., 2014]
    dist, d_l, d_h = np.array([6.2]), np.array([0.7]), np.array([0.7])
    vals.append(dist)
    errs.append((d_l + d_h)/2)
    plt.errorbar(dist, 3, xerr=[d_l, d_h], marker='o', capsize=4, linewidth=2, capthick=2)

    # [Gaia, DR2]
    parallax, p_err = np.array([0.1510]), np.array([0.0413])  # [mas]
    dist = 1 / parallax
    d_l, d_h = dist - 1 / (parallax + p_err), 1 / (parallax - p_err) - dist
    vals.append(dist)
    errs.append((d_l + d_h)/2)
    plt.errorbar(dist, 4, xerr=[d_l, d_h], marker='o', capsize=4, linewidth=2, capthick=2)

    # [Gaia, DR3]
    parallax, p_err = np.array([0.1692]), np.array([0.0262])  # [mas]
    dist = 1 / parallax
    d_l, d_h = dist - 1 / (parallax + p_err), 1 / (parallax - p_err) - dist
    vals.append(dist)
    errs.append((d_l + d_h)/2)
    plt.errorbar(dist, 5, xerr=[d_l, d_h], marker='o', capsize=4, linewidth=2, capthick=2)

    # average value
    vals, errs = np.array(vals), np.array(errs)
    mean = np.average(vals, weights=1 / errs**2)
    std = np.sum(errs**(-2))**(-0.5)
    plt.plot(np.ones(2) * mean, [-0.5, 5.5], color='black', linestyle='dashed', linewidth=2,
             label=f"{mean:.1f} "r"$\pm$"f"{std:.1f} kpc")
    plt.fill_betweenx([-0.5, 5.5], np.ones(2) * (mean - std), np.ones(2) * (mean + std), color='lightgray', alpha=.5)

    # plot settings
    plt.xlabel("Distance, kpc")
    plt.xlim(0, 14)

    plt.yticks([0, 1, 2, 3, 4, 5],
               ["Hjellming, 2000", "Orosz, 2001", "Chaty, 2003", "MacDonald, 2014", "Gaia DR2, 2018", "Gaia DR3, 2022"],
               rotation=15)
    plt.ylim(-0.5, 5.5)
    plt.legend(loc='best')

    plt.tight_layout()
    save_figure("distance_evolution")
    plt.show()
    return


if __name__ == '__main__':
    plot_distance_evolution()
