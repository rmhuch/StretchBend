import matplotlib.pyplot as plt
import numpy as np
from Converter import Constants

def plot_NCs(fig_label, dataDict, eq_coords, water_idx, xy_ranges):
    """Pulls the data needed to make OH vs Natural Charge plots"""
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    dat = dataDict["NaturalCharges"]
    grid_len = len(np.unique(dat[:, 0]))  # use the length of the unique x-values to reshape grid
    if water_idx[0] == 0:  # if monomer
        squareXY = dat.reshape(grid_len, grid_len, 5)
    else:  # if dimer
        squareXY = dat.reshape(grid_len, grid_len, 8)
    cmap1 = plt.get_cmap("Blues_r")  # set all colors for map
    counter1 = 0
    max1 = len(np.argwhere(squareXY[:, 0, 1] < eq_coords[1]))  # the maximum number of HOHs plotted UNDER eq
    cmap2 = plt.get_cmap("Reds")
    counter2 = 1
    max2 = len(np.argwhere(squareXY[:, 0, 1] > eq_coords[1]))  # the maximum number of HOHs plotted OVER eq
    fig, axs = plt.subplots(1, 3, sharex="all", figsize=(25, 8), dpi=216)
    for idx in np.arange(len(squareXY)):  # pull data for one HOH value
        if idx % 2 == 0:  # only plot the even HOH values
            yx = squareXY[idx]  # OH, HOH where OH is fast HOH is slow
            HOH = yx[0, 1]
            HOHdeg = int(np.rint(HOH * (180 / np.pi)))  # convert value for legend
            OhIdx = np.argwhere(yx[:, 0] == eq_coords[0]).squeeze()
            if HOH == eq_coords[1]:  # set the color for the markers
                MFC = 'w'
            elif HOH < eq_coords[1]:  # bend angle is SMALLER than the equilibrium
                MFC = cmap1(float(counter1) / max1)
                counter1 += 1
            elif HOH > eq_coords[1]:  # bend angle is LARGER than the equilibrium
                MFC = cmap2(float(counter2) / max2)
                counter2 += 1
            else:
                raise Exception(f"Can not assign color to {HOH}")
            for i, ax in enumerate(axs):
                # edit OH range to 20% of max ground state wfn
                x_min = np.argmin(np.abs(yx[:, 0] - xy_ranges[0, 0]))
                x_max = np.argmin(np.abs(yx[:, 0] - xy_ranges[0, 1]))
                x_range = yx[x_min:x_max, 0]
                x_eq0 = x_range - eq_coords[0]
                x_ang = Constants.convert(x_range, "angstroms", to_AU=False)
                y = yx[x_min:x_max, water_idx[i]+2]
                delta_y = y - yx[OhIdx, water_idx[i]+2]
                ax.plot(x_ang, y, marker="s", color="k", markerfacecolor=MFC,
                         markersize=10, markeredgewidth=1, linestyle="None",
                         label=r"$\theta_{\mathrm{HOH}}$ = %s" % HOHdeg)
                coeffs = np.polyfit(x_eq0, y, 4)
                f = np.poly1d(coeffs)
                ax.plot(x_ang, f(x_eq0), "--", color=MFC)
                if i == 0:
                    ax.set_ylabel("Natural Charge of O")
                else:
                    ax.set_ylabel("Natural Charge of H")
                ax.set_xlabel(r"$r_{\mathrm{OH}} (\mathrm{\AA})$")
                if i == 1:
                    ax.set_title("Scanned Hydrogen")
                if i == 2:
                    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left')
        else:
            pass
    figname = fig_label + "NBOplot_test2.png"
    plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
    plt.close()
