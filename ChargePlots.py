import os
import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants

def plot_DeltaQvsHOH(fig_label, dataDict, xy_ranges, water_idx, HchargetoPlot=None):
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(12, 8), dpi=216)
    MC = dataDict["MullikenCharges"]  # mulliken charges for every atom
    eq_idx = np.argmin(dataDict["Energies"])
    eq_coords = dataDict["xyData"][eq_idx]
    grid_len = len(np.unique(dataDict["xyData"][:, 0]))  # use the length of the unique x-values to reshape grid
    squareXY_full = dataDict["xyData"].reshape(grid_len, grid_len, 2)
    squareMC_full = MC.reshape(grid_len, grid_len, len(MC[0, :]))
    # cut down the X (data) values (OH) here to only plot within 20% of wfn maximum - decreases # of lines plotted
    cut_X = np.where((xy_ranges[0, 0] < squareXY_full[:, 0, 0]) & (squareXY_full[:, 0, 0] < xy_ranges[0, 1]))[0]
    squareXY = squareXY_full[cut_X, :, :]
    squareMC = squareMC_full[cut_X, :, :]
    allCoeffs = []
    plottedOH = []
    colors = []
    if HchargetoPlot is None:  # define which Hydrogen's Mulliken Charges will be plotted
        plot_idx = water_idx[1]
    elif HchargetoPlot == "average":
        plot_idx = (water_idx[1], water_idx[2])
    else:
        plot_idx = HchargetoPlot
    cmap1 = plt.get_cmap("Blues_r")
    counter1 = 0
    max1 = len(np.argwhere(squareXY[:, 0, 0] < eq_coords[0]))  # the maximum number of OHs plotted UNDER eq
    cmap2 = plt.get_cmap("Reds")
    counter2 = 1
    max2 = len(np.argwhere(squareXY[:, 0, 0] > eq_coords[0]))  # the maximum number of OHs plotted OVER eq
    for idx in np.arange(len(squareXY)):  # pull data for one rOH value
        xy = squareXY[idx]
        charge = squareMC[idx]
        rOH = xy[0, 0]
        rOHang = np.round(Constants.convert(rOH, "angstroms", to_AU=False), 4)  # convert value for legend
        plottedOH.append(rOHang)
        if rOH == eq_coords[0]:
            color = 'k'
            scolor='k'
            mew = 2
            marker = 'o'
            MFC = 'w'
            zord = 100
        elif rOH < eq_coords[0]:  # bend angle is SMALLER than the equilibrium
            marker = "^"
            color = 'k'
            mew = 1
            MFC = cmap1(float(counter1) / max1)
            scolor = MFC
            zord = counter1
            counter1 += 1
        elif rOH > eq_coords[0]:  # bend angle is LARGER than the equilibrium
            marker = "s"
            color = 'k'
            mew = 1
            MFC = cmap2(float(counter2) / max2)
            scolor=MFC
            zord = counter2
            counter2 += 1
        else:
            raise Exception(f"Can not assign color or marker to {rOH}")
        # edit HOH range to 20% of max ground state wfn
        y_min = np.argmin(np.abs(xy[:, 1] - xy_ranges[1, 0]))
        y_max = np.argmin(np.abs(xy[:, 1] - xy_ranges[1, 1]))
        y_range = xy[y_min:y_max, 1]
        y_deg = y_range * (180 / np.pi)
        if type(plot_idx) is int:
            all_charges = charge[:, plot_idx] - MC[eq_idx, plot_idx]  # subtract off MC at eq to plot difference
        elif type(plot_idx) is tuple:
            chargeA = charge[:, plot_idx[0]] - MC[eq_idx, plot_idx[0]]
            chargeB = charge[:, plot_idx[1]] - MC[eq_idx, plot_idx[1]]
            all_charges = np.average(np.column_stack((chargeA, chargeB)), axis=1)
        else:
            raise Exception(f"plot index of {plot_idx} is not defined")
        y_charges = all_charges[y_min:y_max]
        plt.plot(y_deg, y_charges, marker=marker, color=color, markerfacecolor=MFC, markersize=10,
                 markeredgewidth=mew, linestyle="None", zorder=zord, label=r"$r_{\mathrm{OH}}$ = %s" % rOHang)
        colors.append([marker, MFC])
        # fit to a line, and plot
        coefs = np.polyfit(y_range - eq_coords[1], y_charges, 2)
        f = np.poly1d(coefs)
        allCoeffs.append(coefs)
        plt.plot(y_range * (180 / np.pi), f(y_range - eq_coords[1]), "--", color=scolor, zorder=0)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel(r"$ \theta_{\mathrm{HOH}} (^{\circ})$")
    plt.ylabel(r"$\mathcal{Q}_{\mathrm{Mul}}^{(\mathrm{H})} - \mathcal{Q}_{\mathrm{Mul,eq}}^{(\mathrm{H})}  (e)$")
    plt.ylim(-0.2, 0.1)
    plt.tight_layout()
    figname = fig_label + "MCharges_" + f"H{plot_idx}" + "_HOHvsDeltaQ.png"
    plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
    plt.close()
    slopeDat = np.column_stack((plottedOH, allCoeffs, colors))  # save x values in angstroms
    return slopeDat

def plot_DeltaQvsOH(fig_label, dataDict, xy_ranges, water_idx, HchargetoPlot=None):
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(12, 8), dpi=216)
    MC = dataDict["MullikenCharges"]  # mulliken charges for every atom
    eq_idx = np.argmin(dataDict["Energies"])
    eq_coords = dataDict["xyData"][eq_idx]
    grid_len = len(np.unique(dataDict["xyData"][:, 0]))  # use the length of the unique x-values to reshape grid
    sort_idx = np.lexsort((dataDict["xyData"][:, 0], dataDict["xyData"][:, 1]))  # sort by HOH (1) then OH (0)
    resortYX = dataDict["xyData"][sort_idx]  # resort so OH is fast and HOH is slow so same plotting code can be used
    squareYX_full = resortYX.reshape(grid_len, grid_len, 2)
    MC_sort = MC[sort_idx]  # if we re-sort the variables, we have to resort the charges
    squareMC_full = MC_sort.reshape(grid_len, grid_len, len(MC[0, :]))
    # cut down the Y values (HOH) here to only plot within 20% of wfn maximum - decreases # of lines plotted
    cut_Y = np.where((xy_ranges[1, 0] < squareYX_full[:, 0, 1]) & (squareYX_full[:, 0, 1] < xy_ranges[1, 1]))[0]
    squareYX = squareYX_full[cut_Y, :, :]
    squareMC = squareMC_full[cut_Y, :, :]
    allCoeffs = []
    plottedHOH = []
    colors = []
    if HchargetoPlot is None:  # define which Hydrogen's Mulliken Charges will be plotted
        plot_idx = water_idx[1]
    elif HchargetoPlot == "average":
        plot_idx = (water_idx[1], water_idx[2])
    else:
        plot_idx = HchargetoPlot
    cmap1 = plt.get_cmap("Blues_r")
    counter1 = 0
    max1 = len(np.argwhere(squareYX[:, 0, 1] < eq_coords[1]))  # the maximum number of HOHs plotted UNDER eq
    cmap2 = plt.get_cmap("Reds")
    counter2 = 1
    max2 = len(np.argwhere(squareYX[:, 0, 1] > eq_coords[1]))  # the maximum number of HOHs plotted OVER eq
    for idx in np.arange(len(squareYX)):  # pull data for one HOH value
        yx = squareYX[idx]  # OH, HOH where OH is fast HOH is slow
        charge = squareMC[idx]
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
        if type(plot_idx) is int:
            all_charges = charge[:, plot_idx] - MC[eq_idx, plot_idx]  # subtract off MC at eq to plot difference
        elif type(plot_idx) is tuple:
            chargeA = charge[:, plot_idx[0]] - MC[eq_idx, plot_idx[0]]
            chargeB = charge[:, plot_idx[1]] - MC[eq_idx, plot_idx[1]]
            all_charges = np.average(np.column_stack((chargeA, chargeB)), axis=1)
        else:
            raise Exception(f"plot index of {plot_idx} is not defined")
        y_charges = all_charges[y_min:y_max]
        plt.plot(y_ang, y_charges, marker=marker, color=color, markerfacecolor=MFC, markersize=10,
                 markeredgewidth=mew, linestyle="None", zorder=zord, label=r"$\theta_{\mathrm{HOH}}$ = %s" % HOHdeg)
        colors.append([marker, MFC])
        # fit to a line, and plot
        coefs = np.polyfit(y_range - eq_coords[0], y_charges, 4)
        f = np.poly1d(coefs)
        a = np.polyder(f)
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
    plt.ylabel(r"$\mathcal{Q}_{\mathrm{Mul}}^{(\mathrm{H})} - \mathcal{Q}_{\mathrm{Mul,eq}}^{(\mathrm{H})}  (e)$")
    plt.ylim(-0.2, 0.1)
    plt.tight_layout()
    figname = fig_label + "MCharges_" + f"H{plot_idx}" + "_OHvsDeltaQ4.png"
    plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
    plt.close()
    slopeDat = np.column_stack((plottedHOH, allCoeffs, colors))  # save x values in degrees
    return slopeDat

def plotChargeSlopes(fig_label, slopeData, xlabel=None, Hbound=None, HChargetoPlot=None):
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(8, 8), dpi=216)
    x = slopeData[:, 0].astype(float)   # the x argument (either HOH or OH)
    slope = slopeData[:, 4].astype(float)  # the slope (first derivative of the Q plot) **change this if change polyfit
    markers = slopeData[:, -2]  # the marker used for each Q plot
    MFCs = slopeData[:, -1]  # the color of each marker face used in the Q plot
    for i in np.arange(len(x)):
        if markers[i] == "o":
            plt.plot(x[i], slope[i], marker=markers[i], color='k', markeredgewidth=1, markersize=10)
        else:
            plt.plot(x[i], slope[i], marker=markers[i], color='k', markerfacecolor=MFCs[i],
                     markeredgewidth=1, markersize=10)
    # fit to a line, and plot
    x_dat = x - x[int(np.argwhere(markers == "o"))]
    coefs = np.polyfit(x_dat, slope, 4)
    f = np.poly1d(coefs)
    plt.plot(x, f(x_dat), "--", color='k', label=np.round(coefs[3], 8), zorder=-1)
    plt.ylabel(r"Slope of $\Delta \mathcal{Q}$")
    if xlabel == "HOH":
        plt.xlabel(r"$\theta_{\mathrm{HOH}} (^\circ)$")
    elif xlabel == "OH":
        plt.xlabel(r"$r_{\mathrm{OH}} (\mathrm{\AA})$")
    else:
        plt.xlabel(xlabel)
    if xlabel == "HOH":
        if Hbound is None:
            if HChargetoPlot == "average":
                plt.ylim(0, 0.25)
            else:
                plt.ylim(0.225, 0.475)
        elif Hbound:  # if R5 scan
            if HChargetoPlot == 5:   # if plotting r5
                plt.ylim(0, 0.3)
            elif HChargetoPlot == "average":  # if plotting average of two H's
                plt.ylim(-0.1, 0.15)
            else:
                plt.ylim(-0.15, 0.1)
        else:  # r4 scan
            if HChargetoPlot == 4:  # if plotting r4
                plt.ylim(0.25, 0.5)
            elif HChargetoPlot == "average":  # if plotting average of two H's
                plt.ylim(0.0, 0.25)
            else:
                plt.ylim(-0.15, 0.1)
    else:  # if OH slopes
        plt.ylim(-0.15, 0.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_label, dpi=fig.dpi, bbox_inches="tight")


