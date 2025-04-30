import matplotlib as mpl

Linestyles = ['solid', 'dashed', 'dashdot', 'dotted']


def set_plotting_defaults():
    # default figure size
    mpl.rcParams["figure.figsize"] = [7, 5]

    # Set default font size
    mpl.rcParams['font.size'] = 14

    # Grid parameters
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = 'dashed'
    mpl.rcParams['grid.color'] = 'lightgray'

    return


if __name__ == '__main__':
    print('Not for direct use')