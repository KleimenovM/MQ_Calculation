import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from scipy.special import kv

from config.plotting import Tab10
from src.synchrotron_emission import first_synchrotron_function_approximation

from config.plotting import set_plotting_defaults
from config.units import Gauss
from src.electron_spectrum_parametrization import SpectrumParametrization


def test_first_synchrotron_function_approximation():
    set_plotting_defaults()

    t = np.logspace(-4, 2, 1000)
    K53 = kv(5 / 3, t)
    plt.title(r"$t \cdot K_{5/3}(t)$")
    plt.loglog(t, K53, label=r'$K_{5/3}$')
    plt.loglog(t, t * K53, label=r'$t \cdot K_{5/3}$')
    plt.legend(loc=3)
    plt.show()

    return


def first_synchrotron_function(x1):
    t1 = np.logspace(np.log10(x1), 2.2, 1200)
    ln_t1 = np.log(t1)
    k1 = kv(5/3, t1)
    return np.trapezoid(t1 * k1, ln_t1) * x1


def test_first_synchrotron_function():
    x = np.logspace(-4, 2, 1000)

    fsf = np.ones_like(x)

    for i in range(len(x)):
        fsf[i] = first_synchrotron_function(x[i])

    x = np.logspace(-4, 2, 1000)
    fsf_a = first_synchrotron_function_approximation(x)

    set_plotting_defaults()
    gridspec_kw = dict(height_ratios=[2, 1],
                       hspace=0)

    _, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, gridspec_kw=gridspec_kw,
    )

    ax1 = axes[0]
    ax1.loglog(x, fsf_a, label='approximation')
    ax1.loglog(x, fsf, label='exact')
    ax1.legend(loc=3)
    ax1.set_ylim(1e-5, 1e1)
    ax1.set_xlim(1e-4, 1e2)
    ax1.set_ylabel(r'$F(x)$')

    ax2 = axes[1]
    ax2.plot(x, -100 * (fsf_a - fsf) / fsf, label='relative error', color=Tab10[2])
    ax2.set_ylabel(r'$\Delta F,$ %')
    ax2.set_ylim(-0.75, 0.75)
    ax2.legend(loc=3)
    plt.show()

    return


if __name__ == '__main__':
    test_first_synchrotron_function()
