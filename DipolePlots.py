# This script is meant to mimic ChargePlots.py, setting up functions to plot the x & y dipoles vs rOH
import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants

def plot_DipolevsOH(fig_label, dataDict, xy_ranges, DipoletoPlot=None):
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(12, 8), dpi=216)
    Dip = dataDict["RotatedDipoles"]  # x, y, z rotated dipoles
    eq_idx = np.argmin(dataDict["Energies"])
    eq_coords = dataDict["xyData"][eq_idx]
    grid_len = len(np.unique(dataDict["xyData"][:, 0]))  # use the length of the unique x-values to reshape grid
    sort_idx = np.lexsort((dataDict["xyData"][:, 0], dataDict["xyData"][:, 1]))  # sort by HOH (1) then OH (0)
    resortYX = dataDict["xyData"][sort_idx]  # resort so OH is fast and HOH is slow
    squareYX_full = resortYX.reshape(grid_len, grid_len, 2)
    # pull Dipole we want to plot, and resort to match coords
    if DipoletoPlot is "X":
        Dip_sort = Dip[sort_idx, 0]
        eqDip = Dip[eq_idx, 0]
    elif DipoletoPlot is "Y":
        Dip_sort = Dip[sort_idx, 1]
        eqDip = Dip[eq_idx, 1]
    elif DipoletoPlot is "Z":
        Dip_sort = Dip[sort_idx, 2]
        eqDip = Dip[eq_idx, 2]
    elif DipoletoPlot is "Mag":
        DipMag = np.sqrt(Dip[:, 0]**2 + Dip[:, 1]**2 + Dip[:, 2]**2)
        Dip_sort = DipMag[sort_idx]
        eqDip = np.sqrt(Dip[eq_idx, 0]**2 + Dip[eq_idx, 1]**2 + Dip[eq_idx, 2]**2)
    else:
        raise Exception(f"Dipole {DipoletoPlot} can not be plotted.")
    squareDip_full = Dip_sort.reshape(grid_len, grid_len)
    # cut down the Y values (HOH) here to only plot within 20% of wfn maximum - decreases # of lines plotted
    cut_Y = np.where((xy_ranges[1, 0] < squareYX_full[:, 0, 1]) & (squareYX_full[:, 0, 1] < xy_ranges[1, 1]))[0]
    squareYX = squareYX_full[cut_Y, :, :]
    squareDip = squareDip_full[cut_Y, :]
    # set up colleciton lists
    allCoeffs = []
    plottedHOH = []
    colors = []
    # set up color maps
    cmap1 = plt.get_cmap("Blues_r")
    counter1 = 0
    max1 = len(np.argwhere(squareYX[:, 0, 1] < eq_coords[1]))  # the maximum number of HOHs plotted UNDER eq
    cmap2 = plt.get_cmap("Reds")
    counter2 = 1
    max2 = len(np.argwhere(squareYX[:, 0, 1] > eq_coords[1]))  # the maximum number of HOHs plotted OVER eq
    for idx in np.arange(len(squareYX)):  # pull data for one HOH value
        yx = squareYX[idx]  # OH, HOH where OH is fast HOH is slow
        dip = squareDip[idx]
        HOH = yx[0, 1]
        HOHdeg = int(np.rint(HOH * (180 / np.pi)))  # convert value for legend
        plottedHOH.append(HOHdeg)
        if HOH == eq_coords[1]:
            color = 'k'
            scolor='k'
            mew = 2
            marker = 'o'
            MFC = 'w'
            zord = 100
        elif HOH < eq_coords[1]:  # bend angle is SMALLER than the equilibrium
            marker = "^"
            color = 'k'
            mew = 1
            MFC = cmap1(float(counter1) / max1)
            scolor = MFC
            zord = counter1
            counter1 += 1
        elif HOH > eq_coords[1]:  # bend angle is LARGER than the equilibrium
            marker = "s"
            color = 'k'
            mew = 1
            MFC = cmap2(float(counter2) / max2)
            scolor=MFC
            zord = counter2
            counter2 += 1
        else:
            raise Exception(f"Can not assign color or marker to {HOH}")
        # edit OH range to 20% of max ground state wfn
        y_min = np.argmin(np.abs(yx[:, 0] - xy_ranges[0, 0]))
        y_max = np.argmin(np.abs(yx[:, 0] - xy_ranges[0, 1]))
        y_range = yx[y_min:y_max, 0]
        y_ang = Constants.convert(y_range, "angstroms", to_AU=False)
        shiftdip = dip - eqDip
        y_charges = shiftdip[y_min:y_max]
        plt.plot(y_ang, y_charges, marker=marker, color=color, markerfacecolor=MFC, markersize=10,
                 markeredgewidth=mew, linestyle="None", zorder=zord, label=r"$\theta_{\mathrm{HOH}}$ = %s" % HOHdeg)
        colors.append([marker, MFC])
        # fit to a line, and plot
        coefs = np.polyfit(y_range - eq_coords[0], y_charges, 4)
        f = np.poly1d(coefs)
        allCoeffs.append(coefs)
        if HOH == eq_coords[1]:
            plt.plot(Constants.convert(y_range, "angstroms", to_AU=False), f(y_range - eq_coords[0]), "-", color=scolor)
            E_idx = np.argwhere(y_range == eq_coords[0])  # plot the eq/eq point as a black filled circle
            plt.plot(y_ang[E_idx], y_charges[E_idx], marker="o", color=color, markersize=10, linestyle=None, zorder=101)
        else:
            plt.plot(Constants.convert(y_range, "angstroms", to_AU=False), f(y_range - eq_coords[0]), "--",
                     color=scolor)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel(r"$r_{\mathrm{OH}} (\mathrm{\AA})$")
    plt.ylabel(r"$\mu_{%s} - \mu_{%s, eq}$ (A.U.)" % (DipoletoPlot, DipoletoPlot))
    plt.ylim(-0.3, 0.3)
    plt.tight_layout()
    figname = fig_label + "Dip" + DipoletoPlot + "_OHvsDeltaDip.png"
    plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
    plt.close()
    slopeDat = np.column_stack((plottedHOH, allCoeffs, colors))  # save x values in degrees
    return slopeDat

def plotDipSlopes(fig_label, slopeData, DipoletoPlot=None):
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(8, 8), dpi=216)
    x = slopeData[:, 0].astype(float)   # the x argument (either HOH or OH)
    slope = slopeData[:, 4].astype(float)  # the slope (first derivative of the Q plot) **change this if change polyfit
    markers = slopeData[:, -2]  # the marker used for each Q plot
    MFCs = slopeData[:, -1]  # the color of each marker face used in the Q plot
    for i in np.arange(len(x)):
        if markers[i] == "o":
            plt.plot(x[i], slope[i], marker=markers[i], color='k', markeredgewidth=1, markersize=10)
            print(slope[i])
        else:
            plt.plot(x[i], slope[i], marker=markers[i], color='k', markerfacecolor=MFCs[i],
                     markeredgewidth=1, markersize=10)
    # fit to a line, and plot
    x_dat = x - x[int(np.argwhere(markers == "o"))]
    coefs = np.polyfit(x_dat, slope, 4)
    f = np.poly1d(coefs)
    plt.plot(x, f(x_dat), "--", color='k', label=np.round(coefs[3], 8), zorder=-1)
    plt.ylabel(r"Slope of $\mu_{%s}$" % DipoletoPlot)
    plt.xlabel(r"$\theta_{\mathrm{HOH}} (^\circ)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_label, dpi=fig.dpi, bbox_inches="tight")
