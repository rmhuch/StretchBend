import os
import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants

def plot_DeltaQvsHOH(fig_label, dataDict, xy_ranges):
    plt.rcParams.update({'font.size': 20,
                         'legend.fontsize': 18})
    fig = plt.figure(figsize=(12, 8), dpi=216)
    MC = dataDict["MullikenCharges"]  # mulliken charges at every point O H(scan coord) H
    eq_idx = np.argmin(dataDict["Energies"])
    eq_coords = dataDict["xyData"][eq_idx]
    grid_len = len(np.unique(dataDict["xyData"][:, 0]))  # use the length of the unique x-values to reshape grid
    squareXY = dataDict["xyData"].reshape(grid_len, grid_len, 2)
    squareMC = MC.reshape(grid_len, grid_len, 3)
    cmap = plt.get_cmap("seismic")
    counter = 0
    allCoeffs = []
    plottedOH = []
    for idx in np.arange(len(squareXY)):
        # pull data for one rOH value
        xy = squareXY[idx]
        charge = squareMC[idx]
        rOH = xy[0, 0]
        rOHang = np.round(Constants.convert(rOH, "angstroms", to_AU=False), 4)
        # edit OH range to 20% of max ground state wfn
        if rOH < xy_ranges[0, 0] or rOH > xy_ranges[0, 1]:
            pass
        else:
            plottedOH.append(rOHang)
            counter += 1
            # edit HOH range to 20% of max ground state wfn
            y_min = np.argmin(np.abs(xy[:, 1] - xy_ranges[1, 0]))
            y_max = np.argmin(np.abs(xy[:, 1] - xy_ranges[1, 1]))
            y_range = xy[y_min:y_max, 1]
            all_charges = charge[:, 1] - MC[eq_idx, 1]  # subtract off MC at eq to plot difference
            y_charges = all_charges[y_min:y_max]
            if rOH == eq_coords[0]:
                color = 'k'
            else:
                color = cmap(float(counter)/13)
            plt.plot(y_range * (180 / np.pi), y_charges, "o", color=color,
                     label=r"$r_{\mathrm{OH}}$ = %s" % rOHang)
            # fit to a line, and plot
            coefs = np.polyfit(y_range - eq_coords[1], y_charges, 2)
            f = np.poly1d(coefs)
            allCoeffs.append(coefs)
            plt.plot(y_range * (180 / np.pi), f(y_range - eq_coords[1]), "--", color=color)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel(r"$ \theta_{\mathrm{HOH}} (^{\circ})$")
    plt.ylabel(r"$\mathcal{Q}_{\mathrm{Mul}}^{(\mathrm{H})} - \mathcal{Q}_{\mathrm{Mul,eq}}^{(\mathrm{H})}  (e)$")
    plt.tight_layout()
    figname = fig_label + "MCharges_HOHvsDeltaQ.png"
    plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
    plt.close()
    slopeDat = np.column_stack((plottedOH, allCoeffs))  # save x values in angstroms
    return slopeDat

def plot_DeltaQvsOH(fig_label, dataDict, xy_ranges):
    plt.rcParams.update({'font.size': 20,
                         'legend.fontsize': 18})
    fig = plt.figure(figsize=(12, 8), dpi=216)
    MC = dataDict["MullikenCharges"]  # mulliken charges at every point O H(scan coord) H
    eq_idx = np.argmin(dataDict["Energies"])
    eq_coords = dataDict["xyData"][eq_idx]
    grid_len = len(np.unique(dataDict["xyData"][:, 0]))  # use the length of the unique x-values to reshape grid
    sort_idx = np.lexsort((dataDict["xyData"][:, 1], dataDict["xyData"][:, 1]))
    resortYX = dataDict["xyData"][sort_idx]  # resort so OH is fast and HOH is slow so same plotting code can be used
    squareYX = resortYX.reshape(grid_len, grid_len, 2)
    MC_sort = MC[sort_idx]  # if we re-sort the variables, we have to resort the charges
    squareMC = MC_sort.reshape(grid_len, grid_len, 3)
    cmap = plt.get_cmap("seismic")
    counter = 0
    allCoeffs = []
    plottedHOH = []
    for idx in np.arange(len(squareYX)):
        # pull data for one HOH value
        xy = squareYX[idx]  # OH, HOH where OH is fast HOH is slow
        charge = squareMC[idx]
        HOH = xy[0, 1]
        HOHdeg = int(np.rint(HOH * (180 / np.pi)))
        # edit HOH range to 20% of max ground state wfn
        if HOH < xy_ranges[1, 0] or HOH > xy_ranges[1, 1]:
            pass
        else:
            plottedHOH.append(HOHdeg)
            counter += 1
            # edit OH range to 20% of max ground state wfn
            y_min = np.argmin(np.abs(xy[:, 0] - xy_ranges[0, 0]))
            y_max = np.argmin(np.abs(xy[:, 0] - xy_ranges[0, 1]))
            y_range = xy[y_min:y_max, 0]
            all_charges = charge[:, 1] - MC[eq_idx, 1]  # subtract off MC at eq to plot difference
            y_charges = all_charges[y_min:y_max]
            if HOH == eq_coords[1]:
                color = 'k'
            else:
                color = cmap(float(counter)/13)
            plt.plot(Constants.convert(y_range, "angstroms", to_AU=False), y_charges, "o", color=color,
                     label=r"$\theta_{\mathrm{HOH}}$ = %s" % HOHdeg)
            # fit to a line, and plot
            coefs = np.polyfit(y_range-eq_coords[0], y_charges, 2)
            f = np.poly1d(coefs)
            allCoeffs.append(coefs)
            plt.plot(Constants.convert(y_range, "angstroms", to_AU=False), f(y_range - eq_coords[0]), "--", color=color)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel(r"$r_{\mathrm{OH}} (\mathrm{\AA})$")
    plt.ylabel(r"$\mathcal{Q}_{\mathrm{Mul}}^{(\mathrm{H})} - \mathcal{Q}_{\mathrm{Mul,eq}}^{(\mathrm{H})}  (e)$")
    plt.tight_layout()
    figname = fig_label + "MCharges_OHvsDeltaQ.png"
    plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
    plt.close()
    slopeDat = np.column_stack((plottedHOH, allCoeffs))  # save x values in degrees
    return slopeDat

def plotChargeSlopes(fig_label, slopeData, xlabel=None):
    plt.rcParams.update({'font.size': 20,
                         'legend.fontsize': 18})
    fig = plt.figure(figsize=(8, 8), dpi=216)
    x = slopeData[:, 0]
    slope = slopeData[:, 2]
    cmap = plt.get_cmap("seismic")
    for i in np.arange(len(x)):
        if i == 5:
            color = "k"
        else:
            color = cmap(float(i+1) / 13)
        plt.plot(x[i], slope[i], "o", color=color)
    plt.ylabel(r"Slope of $\Delta \mathcal{Q}$")
    if xlabel == "HOH":
        plt.xlabel(r"$\theta_{\mathrm{HOH}} (^\circ)$")
    elif xlabel == "OH":
        plt.xlabel(r"$r_{\mathrm{OH}} (\mathrm{\AA})$")
    else:
        plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(fig_label, dpi=fig.dpi, bbox_inches="tight")


