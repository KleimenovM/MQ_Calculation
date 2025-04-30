import os
import pickle

import matplotlib.pyplot as plt

from config.plotting import save_figure, Tab10, set_plotting_defaults
from config.settings import SPECTRUM_DIR
from spectrum.gamma_ray_measurements import gamma_ray_data


def test_ic_spectrum_modification():
    data = pickle.load(open(os.path.join(SPECTRUM_DIR, "UHE_spectrum_corrected.pck"), "rb"))
    names, e, f_cor, f_l_cor, f_p_cor, e_l, e_p = data
    names, e, f, f_l, f_p, e_l, e_p = gamma_ray_data()

    set_plotting_defaults()

    plt.errorbar((1e12, 1e12), (1, 1), label='obs. / intr.', linestyle='None', fmt='')
    for i, name in enumerate(names):
        plt.errorbar(e[i], f[i], xerr=[e_l[i], e_p[i]], yerr=[f_l[i], f_p[i]], fmt='s',
                     linestyle='None', uplims=f_p[i] <= 0,
                     color=Tab10[i], alpha=.5, label=f'{names[i]}')

    plt.errorbar((1e12, 1e12), (1, 1), label=' ', linestyle='None', fmt='')
    for i, name in enumerate(names):
        plt.errorbar(e[i], f_cor[i], xerr=[e_l[i], e_p[i]], yerr=[f_l_cor[i], f_p_cor[i]],
                     fmt='o', linestyle='None', uplims=f_p[i] <= 0,
                     color=Tab10[i], label=f' ')

    plt.legend(loc=2, ncol=2)

    plt.xscale('log')
    plt.xlim(1e9, 1e15)
    plt.xlabel("Energy, eV")

    plt.yscale('log')
    plt.ylim(5e-14, 5e-10)
    plt.ylabel(r"Flux, $\mathrm{erg~cm^{-2}~s^{-1}}$")

    plt.tight_layout()

    # save_figure("spectrum_UHE")
    plt.show()
    return


if __name__ == '__main__':
    test_ic_spectrum_modification()
