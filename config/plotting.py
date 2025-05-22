import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('default')

from config.settings import PICS_DIR

Linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

Tab10 = mpl.color_sequences["tab10"]
Tab20 = mpl.color_sequences["tab20"]

royalblue_palette = [
    '#6A89FF',  # Bright blue — noticeable, but not too light
    '#4169E1',  # Base RoyalBlue
    '#1F3A8A'   # Deep blue — high contrast, still blue (not blackish)
]

orangered_palette = [
    '#FFA07A',  # LightSalmon (brighter variant)
    '#FF4500',  # Orangered (base color)
    '#CC3700',  # Darker Orangered
    '#992800'   # Even darker shade
]


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
    plt.savefig(os.path.join(PICS_DIR, f"{title}.pdf"), dpi=600)
    return


if __name__ == '__main__':
    print('Not for direct use')