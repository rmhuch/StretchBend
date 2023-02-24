import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants


def calc_qDonor(coordies, mus):
    """Calculates the Charge of the donor water for a specific HOH angle OF THE DIMER given the
    equilibirum coordinates at that angle and the dipole"""
    # Acceptor (0, 1, 2) Donor (3, 4-free, 5-bound)
    Xa = coordies[1, 0] + coordies[2, 0] - 2 * coordies[0, 0]  # the X coordinate of the Acceptor
    Za = coordies[1, 2] + coordies[2, 2] - 2 * coordies[0, 2]  # the Z coordinate of the Acceptor
    Xd = (0.59349 * coordies[5, 0]) + (0.40650 * coordies[4, 0]) - coordies[3, 0]  # ratios in dipole distribution excel
    Zd = (0.59349 * coordies[5, 2]) + (0.40650 * coordies[4, 2]) - coordies[3, 2]
    qD = (mus[0] - (mus[2] * Xa / Za)) / (Xd - (Xa * Zd / Za))
    return qD

def pull_HOHpltDat(dataDict, xy_ranges, water_idx, ComptoPlot):
    """Pulls the data needed to make OH vs Delta MU plots, keyed by HOH values"""
    sort_idx = np.lexsort((dataDict["xyData"][:, 0], dataDict["xyData"][:, 1]))  # sort by HOH (1) then OH (0)
    resortYX = dataDict["xyData"][sort_idx]  # resort so OH is fast and HOH is slow
    eq_idx = np.argmin(dataDict["Energies"][sort_idx])  # eq_idx based on YX sort
    eq_coords = resortYX[eq_idx]  # [HOH, OH] atomic units
    grid_len = len(np.unique(dataDict["xyData"][:, 0]))  # use the length of the unique x-values to reshape grid
    squareYX_full = resortYX.reshape(grid_len, grid_len, 2)  # num of HOH x num of OH
    DipYX = dataDict["RotatedDipoles"][sort_idx]  # x, y, z rotated dipoles in the yx order
    squareDip_full = DipYX.reshape(grid_len, grid_len, 3)  # num of HOH x num of OH x XYZ comps
    OhVecs = dataDict["RotatedCoords"][:, water_idx[1], :] - dataDict["RotatedCoords"][:, water_idx[0], :]
    OhVecsYX = OhVecs[sort_idx]
    squareOhVecs_full = OhVecsYX.reshape(grid_len, grid_len, 3)  # num of HOH x num of OH x XYZ comps
    # cut down the Y values (HOH) here to only plot within 20% of wfn maximum - decreases # of lines plotted
    cut_Y = np.where((xy_ranges[1, 0] < squareYX_full[:, 0, 1]) & (squareYX_full[:, 0, 1] < xy_ranges[1, 1]))[0]
    squareYX = squareYX_full[cut_Y, :, :]
    squareOHVecs = squareOhVecs_full[cut_Y, :, :]
    squareDip = squareDip_full[cut_Y, :, :]
    print(ComptoPlot)
    if ComptoPlot == "X":  # assign idx number based on `ComptoPlot`
        c = 0
    elif ComptoPlot == "Y":
        c = 1
    elif ComptoPlot == "Z":
        c = 2
    elif ComptoPlot == "Mag":
        c = "Mag"
    else:
        raise Exception(f"Component {ComptoPlot} is not defined")
    # pull all coords
    sCoords = dataDict["RotatedCoords"][sort_idx]
    if water_idx[0] == 0:  # if monomer
        sqCoords = sCoords.reshape(grid_len, grid_len, 3, 3)
        coordInds = [0, 1]  # the x & y are nonzero in the monomer
    else:  # if dimer
        sqCoords = sCoords.reshape(grid_len, grid_len, 6, 3)
        coordInds = [0, 2]  # the x & z are nonzero in the dimer
    squareCoords = sqCoords[cut_Y, :, :, :]
    cmap1 = plt.get_cmap("Blues_r")  # set all colors for map
    counter1 = 0
    max1 = len(np.argwhere(squareYX[:, 0, 1] < eq_coords[1]))  # the maximum number of HOHs plotted UNDER eq
    cmap2 = plt.get_cmap("Reds")
    counter2 = 1
    max2 = len(np.argwhere(squareYX[:, 0, 1] > eq_coords[1]))  # the maximum number of HOHs plotted OVER eq
    PltDat = dict()
    for idx in np.arange(len(squareYX)):  # pull data for one HOH value
        yx = squareYX[idx]  # OH, HOH where OH is fast HOH is slow
        HOH = yx[0, 1]
        carts = squareCoords[idx]
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
        OhIdx = np.argwhere(yx[:, 0] == eq_coords[0]).squeeze()
        if water_idx[0] == 0:  # monomer
            FC = abs(squareDip[idx, OhIdx, 1] / (2 * yx[OhIdx, 0] * np.cos(HOH / 2)))
        elif water_idx[0] > 0:  # dimer
            eqCoords = squareCoords[idx, OhIdx, :, :]  # (6, 3)
            qD = calc_qDonor(eqCoords, squareDip[idx, OhIdx, :])
            if water_idx[1] == 4:  # free
                FC = qD * 0.40650
            elif water_idx[1] == 5:  # bound
                FC = qD * 0.59349
            else:
                raise Exception(f"Can not determine H-type for water index {water_idx}")
        else:
            raise Exception(f"Can not describe Fixed Charge for {water_idx} water index")
        if type(c) == str:
            comp = np.sqrt((FC * squareOHVecs[idx, :, 0]) ** 2 + (FC * squareOHVecs[idx, :, 1]) ** 2
                           + (FC * squareOHVecs[idx, :, 2]) ** 2)
            eqComp = comp[OhIdx]
            # for the dipole magnitude, we will rotate the OH vector to the X coordinate
            cosT = squareOHVecs[idx, :, coordInds[0]] / yx[:, 0]
            sinT = squareOHVecs[idx, :, coordInds[1]] / yx[:, 0]
            dip = (squareDip[idx, :, coordInds[0]] * cosT) + (squareDip[idx, :, coordInds[1]] * sinT)
            eqDip = dip[OhIdx]
        else:
            comp = FC * squareOHVecs[idx, :, c]
            eqComp = comp[OhIdx]
            dip = squareDip[idx, :, c]
            eqDip = dip[OhIdx]
        DC = eqDip / np.linalg.norm(squareOHVecs[idx, OhIdx, :])
        HOHdeg = int(np.rint(HOH * (180 / np.pi)))  # convert value for legend
        print(f"For angle {HOHdeg}: FC - {eqComp}, Dip - {eqDip}")
        # edit OH range to 20% of max ground state wfn
        x_min = np.argmin(np.abs(yx[:, 0] - xy_ranges[0, 0]))
        x_max = np.argmin(np.abs(yx[:, 0] - xy_ranges[0, 1]))
        x_range = yx[x_min:x_max, 0]
        x_eq0 = x_range - eq_coords[0]
        x_ang = Constants.convert(x_range, "angstroms", to_AU=False)
        delta_y = comp[x_min:x_max] - eqComp  # change in OH from eq
        y_charges = delta_y
        coefsFC = np.polyfit(x_eq0, y_charges, 4)
        shiftdip = dip - eqDip
        y_dips = shiftdip[x_min:x_max]
        coefsDip = np.polyfit(x_eq0, y_dips, 4)
        if type(c) == str:  # this creates the data sets that plot the linear dipole on the magnitude plots
            if water_idx[0] == 0:  # monomer
                y_deriv = 0.19787 * x_eq0
                coefsDeriv = [0, 0, 0, 0.19787]
            elif water_idx[1] == 4:  # free OH
                y_deriv = 0.17709 * x_eq0
                coefsDeriv = [0, 0, 0, 0.17709]
            elif water_idx[1] == 5:
                y_deriv = 0.59393 * x_eq0
                coefsDeriv = [0, 0, 0, 0.59393]
        else:
            y_deriv = 0
            coefsDeriv = 0
        degDat = {"x_SI": x_ang,  # OH values to be plotted on X-axis
                  "x_eq0": x_eq0,  # OH values shifted so eq is 0 (for expansion)
                  "FC": FC,  # the calculated Fixed Charge for this HOH
                  "y_charges": y_charges,  # FC to be plotted on Y-axis
                  "y_dips": y_dips,  # Dipole moments to be plotted on Y-axis/ If "Mag" then Magnitude rotated to OH
                  "y_deriv": y_deriv,  # Dipole Derivatives based on FD
                  "DerivCoeffs": coefsDeriv,  # the slope for the dipole derivative
                  "DipCharge": DC,  # the charge of the equilibrium dipole
                  "FcCoeffs": coefsFC,  # polyfit coefs of FC line
                  "DipCoeffs": coefsDip,  # polyfit coefs of Dipole line
                  "MFC": MFC}  # color to be used for the marker
        PltDat[HOHdeg] = degDat
    return PltDat

def pull_OHpltDat(dataDict, xy_ranges, water_idx, ComptoPlot):
    """Pulls the data needed to make HOH vs Delta Mu plots, keyed by the OH values."""
    eq_idx = np.argmin(dataDict["Energies"])
    eq_coords = dataDict["xyData"][eq_idx]
    grid_len = len(np.unique(dataDict["xyData"][:, 0]))  # use the length of the unique x-values to reshape grid
    squareXY_full = dataDict["xyData"].reshape(grid_len, grid_len, 2)
    squareDip_full = dataDict["RotatedDipoles"].reshape(grid_len, grid_len, 3)
    OhVecs = dataDict["RotatedCoords"][:, water_idx[1], :] - dataDict["RotatedCoords"][:, water_idx[0], :]
    squareOhVecs_full = OhVecs.reshape(grid_len, grid_len, 3)  # num of HOH x num of OH x XYZ comps
    # cut down the X (data) values (OH) here to only plot within 20% of wfn maximum - decreases # of lines plotted
    cut_X = np.where((xy_ranges[0, 0] < squareXY_full[:, 0, 0]) & (squareXY_full[:, 0, 0] < xy_ranges[0, 1]))[0]
    squareXY = squareXY_full[cut_X, :, :]
    squareDip = squareDip_full[cut_X, :, :]
    squareOhVecs = squareOhVecs_full[cut_X, :, :]
    if ComptoPlot == "X":  # assign idx number based on `ComptoPlot`
        c = 0
    elif ComptoPlot == "Y":
        c = 1
    elif ComptoPlot == "Z":
        c = 2
    elif ComptoPlot == "Mag":
        c = "Mag"
    else:
        raise Exception(f"Component {ComptoPlot} is not defined")
    if water_idx[0] > 0:  # only if plotting dimer
        sCoords = dataDict["RotatedCoords"].reshape(grid_len, grid_len, 6, 3)
        squareCoords = sCoords[cut_X, :, :, :]
    cmap1 = plt.get_cmap("Blues_r")
    counter1 = 0
    max1 = len(np.argwhere(squareXY[:, 0, 0] < eq_coords[0]))  # the maximum number of OHs plotted UNDER eq
    cmap2 = plt.get_cmap("Reds")
    counter2 = 1
    max2 = len(np.argwhere(squareXY[:, 0, 0] > eq_coords[0]))  # the maximum number of OHs plotted OVER eq
    PltDat = dict()
    for idx in np.arange(len(squareXY)):  # pull data for one rOH value
        xy = squareXY[idx]  # OH, HOH where HOH is fast and OH is slow
        OH = xy[0, 0]
        if OH == eq_coords[0]:  # set the color for the markers
            MFC = 'w'
        elif OH < eq_coords[0]:  # bond length is SMALLER than the equilibrium
            MFC = cmap1(float(counter1) / max1)
            counter1 += 1
        elif OH > eq_coords[0]:  # bond length is LARGER than the equilibrium
            MFC = cmap2(float(counter2) / max2)
            counter2 += 1
        else:
            raise Exception(f"Can not assign color to {OH}")
        HohIdx = np.argwhere(xy[:, 1] == eq_coords[1]).squeeze()
        if water_idx[0] == 0:  # monomer
            FC = abs(squareDip[idx, HohIdx, 1] / (2 * OH * np.cos(xy[HohIdx, 1] / 2)))
        elif water_idx[0] > 0:  # dimer
            eqCoords = squareCoords[idx, HohIdx, :, :]
            qD = calc_qDonor(eqCoords, squareDip[idx, HohIdx, :])
            if water_idx[1] == 4:
                FC = qD * 0.40650
            elif water_idx[1] == 5:
                FC = qD * 0.59349
            else:
                raise Exception(f"Can not determine H-type for water index{water_idx}")
        else:
            raise Exception(f"Can not describe Fixed Charge for {water_idx} water index")
        if type(c) == str:
            comp = np.sqrt((FC * squareOhVecs[idx, :, 0]) ** 2 + (FC * squareOhVecs[idx, :, 1]) ** 2
                           + (FC * squareOhVecs[idx, :, 2]) ** 2)
            eqComp = comp[HohIdx]
            dip = np.sqrt(squareDip[idx, :, 0] ** 2 + squareDip[idx, :, 1] ** 2 + squareDip[idx, :, 2] ** 2)
            eqDip = dip[HohIdx]
        else:
            comp = FC * squareOhVecs[idx, :, c]
            eqComp = comp[HohIdx]
            dip = squareDip[idx, :, c]
            eqDip = dip[HohIdx]
        OHang = np.round(Constants.convert(OH, "angstroms", to_AU=False), 3)
        # edit HOH range to 20% of max ground state wfn
        x_min = np.argmin(np.abs(xy[:, 1] - xy_ranges[1, 0]))
        x_max = np.argmin(np.abs(xy[:, 1] - xy_ranges[1, 1]))
        x_range = xy[x_min:x_max, 1]
        x_eq0 = x_range - eq_coords[1]
        x_deg = x_range * (180 / np.pi)
        delta_y = comp[x_min:x_max] - eqComp
        y_charges = delta_y
        coefsFC = np.polyfit(x_eq0, y_charges, 4)
        shiftdip = dip - eqDip
        y_dips = shiftdip[x_min:x_max]
        coefsDip = np.polyfit(x_eq0, y_dips, 4)
        angDat = {"x_SI": x_deg,  # HOH values to be plotted on X-axis
                  "x_eq0": x_eq0,  # HOH values shifted so eq is 0 (for expansion)
                  "y_charges": y_charges,  # FC to be plotted on Y-axis
                  "y_dips": y_dips,  # Dipole moments to be plotted on Y-axis
                  "FcCoeffs": coefsFC,  # polyfit coefs of FC line
                  "DipCoeffs": coefsDip,  # polyfit coefs of Dipole line
                  "MFC": MFC}  # color to be used for the marker
        PltDat[OHang] = angDat
    return PltDat

def plot_FCDipvsOH(fig_label, dataDict, xy_ranges, water_idx, ComptoPlot=None, EQonly=False, Xaxis="OH"):
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(12, 8), dpi=216)
    if Xaxis == "OH":
        PltDat = pull_HOHpltDat(dataDict, xy_ranges, water_idx, ComptoPlot)
        vals = [96, 104, 112]
        Xlabel = r"$r_{\mathrm{OH}} (\mathrm{\AA})$"
    elif Xaxis == "HOH":
        PltDat = pull_OHpltDat(dataDict, xy_ranges, water_idx, ComptoPlot)
        if water_idx[0] == 0:
            vals = [0.911, 0.961, 1.011]
        elif water_idx[1] == 4:
            vals = [0.91, 0.96, 1.01]
        elif water_idx[1] == 5:
            vals = [0.918, 0.968, 1.018]
        else:
            raise Exception(f"Can not determine OH vals for water idx {water_idx}")
        Xlabel = r"$\theta_{\mathrm{HOH}} (^{\circ})$"
    # a dictionary keyed by HOH vals holding all data needed to make the plots
    else:
        raise Exception(f"Can not determine data for X-axis {Xaxis}")
    if EQonly:
        eqDict = PltDat[vals[1]]
        # plot FC points
        plt.plot(eqDict["x_SI"], eqDict["y_charges"], marker="s", color="k", markerfacecolor="forestgreen",
                 markersize=10, markeredgewidth=1, linestyle="None", label="Fixed Charge Model")
        f = np.poly1d(eqDict["FcCoeffs"])
        plt.plot(eqDict["x_SI"], f(eqDict["x_eq0"]), "--", color="k")
        # plot Dipole points
        if ComptoPlot == "Mag":
            plt.plot(eqDict["x_SI"], eqDict["y_deriv"], color="fuchsia", linewidth=3.0, label="Linear Dipole")
        plt.plot(eqDict["x_SI"], eqDict["y_dips"], marker="o", color="k", markerfacecolor="rebeccapurple",
                 markersize=10, markeredgewidth=1, linestyle="None", label="Full Dipole")
        f1 = np.poly1d(eqDict["DipCoeffs"])
        plt.plot(eqDict["x_SI"], f1(eqDict["x_eq0"]), "-", color="k")
    else:
        for deg in PltDat:  # pull data for one HOH value...
            if deg == vals[0] or deg == vals[1] or deg == vals[2]:
                degDict = PltDat[deg]
                # plot all the FC plots
                plt.plot(degDict["x_SI"], degDict["y_charges"], marker="s", color="k", markerfacecolor=degDict["MFC"],
                         markersize=10, markeredgewidth=1, linestyle="None",
                         label=np.round(PltDat[deg]["FcCoeffs"][3], 8))
                f = np.poly1d(degDict["FcCoeffs"])
                plt.plot(degDict["x_SI"], f(degDict["x_eq0"]), "--", color=degDict["MFC"])
                # plot all the Dipole plots
                if ComptoPlot == "Mag":
                    plt.plot(degDict["x_SI"], degDict["y_deriv"], color=degDict["MFC"],
                             label=np.round(PltDat[deg]["DipCoeffs"][3], 8))
                else:
                    plt.plot(degDict["x_SI"], degDict["y_dips"], marker="o", color="k", markerfacecolor=degDict["MFC"],
                             markersize=10, markeredgewidth=1, linestyle="None",
                             label=np.round(PltDat[deg]["DipCoeffs"][3], 8))
                    f1 = np.poly1d(degDict["DipCoeffs"])
                    plt.plot(degDict["x_SI"], f1(degDict["x_eq0"]), "-", color=degDict["MFC"])
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel(Xlabel)
    plt.ylabel(r"$\Delta \mu_{%s}$" % ComptoPlot)
    plt.ylim(-0.25, 0.25)
    plt.tight_layout()
    if EQonly:
        figname = fig_label + "DeltaMu_" + ComptoPlot + "_" + Xaxis + "DipFCplot_EQonlyDeriv2.png"
    else:
        figname = fig_label + "DeltaMu_" + ComptoPlot + "_" + Xaxis + "DipFCplot_test2Deriv.png"
    plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
    plt.close()

def plotFCvsHOH(fig_label, dataDict, xy_ranges, water_idx, ComptoPlot=None):
    """THIS ISN'T QUITE RIGHT YET, KEEP THINKING OF A BETTER WAY TO REPRESENT THIS"""
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(12, 8), dpi=216)
    PltDat = pull_HOHpltDat(dataDict, xy_ranges, water_idx, ComptoPlot)
    Xlabel = r"$\theta_{\mathrm{HOH}} (^{\circ})$"
    for deg in PltDat:  # pull data for one HOH value...
        degDict = PltDat[deg]
        # plot all the FC plots
        plt.plot(deg, degDict["FC"], marker="s", color="k", markerfacecolor=degDict["MFC"],
                 markersize=10, markeredgewidth=1, linestyle="None")
        # plot all the Dipole plots
        plt.plot(deg, degDict["DipCharge"], marker="o", color="k", markerfacecolor=degDict["MFC"],
                 markersize=6, markeredgewidth=1, linestyle="None")
    plt.xlabel(Xlabel)
    plt.ylabel("Charge")
    # plt.ylim(-0.25, 0.25)
    plt.tight_layout()
    figname = fig_label + "FC_HOHplot_test1.png"
    plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
    plt.close()

def plotFCDipSlopes(fig_label, dataDict, xy_ranges, water_idx, ComptoPlot=None):
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(8, 8), dpi=216)
    PltDat = pull_HOHpltDat(dataDict, xy_ranges, water_idx, ComptoPlot)
    x_dat = []
    FCslopes = []
    Dipslopes = []
    for deg in PltDat:
        x_dat.append(deg)
        FCslopes.append(PltDat[deg]["FcCoeffs"][3])
        Dipslopes.append(PltDat[deg]["DipCoeffs"][3])
        plt.plot(deg, PltDat[deg]["FcCoeffs"][3], marker="s", color='k', markerfacecolor=PltDat[deg]["MFC"],
                 markeredgewidth=1, markersize=10)
        plt.plot(deg, PltDat[deg]["DipCoeffs"][3], marker="o", color='k', markerfacecolor=PltDat[deg]["MFC"],
                 markeredgewidth=1, markersize=10)
    x_dat = np.array(x_dat)
    FCslopes = np.array(FCslopes)
    Dipslopes = np.array(Dipslopes)
    # fit to a line, and plot
    x_shift = x_dat - 104  # shift so eq is at x=0 for expansion
    coefs = np.polyfit(x_shift, FCslopes, 4)
    coefs1 = np.polyfit(x_shift, Dipslopes, 4)
    f = np.poly1d(coefs)
    f1 = np.poly1d(coefs1)
    plt.plot(x_dat, f(x_shift), "--", color='k', label=np.round(coefs[3], 8), zorder=-1)
    plt.plot(x_dat, f1(x_shift), "-", color='k', label=np.round(coefs1[3], 8), zorder=-2)
    if water_idx[1] == 5 and ComptoPlot == "X":
        plt.ylim(-0.7, 0)
    elif water_idx[1] == 5 and ComptoPlot == "Mag":
        plt.ylim(0, 0.7)
    else:
        plt.ylim(-0.25, 0.4)
    plt.ylabel(r"Slope of $\Delta \mu_{%s}$" % ComptoPlot)
    plt.xlabel(r"$\theta_{\mathrm{HOH}} (^\circ)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_label, dpi=fig.dpi, bbox_inches="tight")
