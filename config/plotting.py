import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt

from config.settings import PICS_DIR

Linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

Tab10 = mpl.color_sequences["tab10"]
Tab20 = mpl.color_sequences["tab20"]


def set_plotting_defaults():
    # default figure size
    mpl.rcParams["figure.figsize"] = [7, 5]

    # Set default font size
    mpl.rcParams['font.size'] = 12

    # Grid parameters
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = 'dashed'
    mpl.rcParams['grid.color'] = 'lightgray'

    return


def save_figure(title):
    plt.savefig(os.path.join(PICS_DIR, f"{title}.png"), dpi=600, transparent=True)
    plt.savefig(os.path.join(PICS_DIR, f"{title}.pdf"))
    return


if __name__ == '__main__':
    print('Not for direct use')