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
    qD = (mus[0] - ((mus[2] * Xa) / Za)) / (Xd - ((Xa * Zd) / Za))
    return qD

def calc_qAcceptor(coordies, mus, qD):
    """Calculates the Charge of the acceptor water for a specific HOH angle OF THE DIMER given the
    equilibirum coordinates at that angle and the dipole"""
    # Acceptor (0, 1, 2) Donor (3, 4-free, 5-bound)
    Za = coordies[1, 2] + coordies[2, 2] - 2 * coordies[0, 2]  # the Z coordinate of the Acceptor
    Zd = (0.59349 * coordies[5, 2]) + (0.40650 * coordies[4, 2]) - coordies[3, 2]
    qA = (mus[2] - (Zd * qD)) / Za
    return qA

def monomer_FC(yx, squareCoords, Dip, ComptoPlot, FC):
    # This will rotate and calculate the FC for ONE HOH angle at a time...
    OHVecs = squareCoords[:, 1, :] - squareCoords[:, 0, :]
    # calculate the SECOND OH vector, needed to calculate FIXED CHARGE DIPOLE
    OH2Vecs = squareCoords[:, 2, :] - squareCoords[:, 0, :]
    # ROTATE FC comps and Dip so (scanned) OH is on X coordinate
    cosT = OHVecs[:, 0] / yx[:, 0]
    sinT = OHVecs[:, 1] / yx[:, 0]
    rotX = (OHVecs[:, 0] * cosT) + (OHVecs[:, 1] * sinT)
    rotY = (OHVecs[:, 1] * cosT) - (OHVecs[:, 0] * sinT)
    dipX = (Dip[:, 0] * cosT) + (Dip[:, 1] * sinT)
    dipY = (Dip[:, 1] * cosT) - (Dip[:, 0] * sinT)
    # rotate OH2 so in same axis system as OH1 for FC dipole calculation
    # note: OH2 should not change as a function of OH because it is fixed in the scan (i.e. rotX2/Y2 array should be the same value repeated)
    rotX2 = (OH2Vecs[:, 0] * cosT) + (OH2Vecs[:, 1] * sinT)
    rotY2 = (OH2Vecs[:, 1] * cosT) - (OH2Vecs[:, 0] * sinT)
    if ComptoPlot == "Mag":
        comp = FC * np.sqrt((rotX + rotX2) ** 2 + (rotY + rotY2) ** 2)
        dip = np.sqrt(dipX ** 2 + dipY ** 2)
    else:
        if ComptoPlot == "X":
            comp = FC * (rotX + rotX2)
            dip = dipX
        elif ComptoPlot == "Y":
            comp = FC * (rotY + rotY2)
            dip = dipY
        else:
            raise Exception(f"ComptoPlot {ComptoPlot} unrecognized...")
    return comp, dip, np.array((cosT, sinT))  # need to pull out cos/sin rotations to apply to deriv

def dimer_FC(yx, squareCoords, Dip, water_idx, ComptoPlot, qD, qA):
    # For the dimer, we need to calculate the dipole w/ the fixed charge using the charge on each atom and the coordinates of each atom.
    # This will rotate and calculate the FC for ONE HOH angle at a time...
    # Acceptor (0, 1, 2) Donor (3, 4-free, 5-bound)
    OHVecs = squareCoords[:, 5, :] - squareCoords[:, 3, :]  # BOUND
    OH2Vecs = squareCoords[:, 4, :] - squareCoords[:, 3, :]  # FREE
    OOVecs = squareCoords[:, 0, :] - squareCoords[:, 3, :]
    OHa1Vecs = squareCoords[:, 1, :] - squareCoords[:, 3, :]
    OHa2Vecs = squareCoords[:, 2, :] - squareCoords[:, 3, :]
    # in this arrangement the y-component is 0, so we calculated fixed dipole charges now.
    muZ = (OHa1Vecs[:, 2] + OHa2Vecs[:, 2] - (2 * OOVecs[:, 2])) * qA + \
          ((0.59349 * OHVecs[:, 2]) + (0.40650 * OH2Vecs[:, 2])) * qD
    muX = (OHa1Vecs[:, 0] + OHa2Vecs[:, 0] - (2 * OOVecs[:, 0])) * qA + \
          ((0.59349 * OHVecs[:, 0]) + (0.40650 * OH2Vecs[:, 0])) * qD
    # ROTATE FC comps and Dip so (scanned) OH is on X coordinate
    if water_idx[1] == 5:  # BOUND
        cosT = OHVecs[:, 0] / yx[:, 0]
        sinT = OHVecs[:, 2] / yx[:, 0]
    elif water_idx[1] == 4:  # FREE
        cosT = OH2Vecs[:, 0] / yx[:, 0]
        sinT = OH2Vecs[:, 2] / yx[:, 0]
    else:
        raise Exception(f"Can not compute rotation vector for scanned H position {water_idx[1]}")
    dipX = (Dip[:, 0] * cosT) + (Dip[:, 2] * sinT)
    dipZ = (Dip[:, 2] * cosT) - (Dip[:, 0] * sinT)
    dipY = Dip[:, 1]
    rotMuX = (muX * cosT) + (muZ * sinT)
    rotMuZ = (muZ * cosT) - (muX * sinT)
    # # rotate OH2 so in same axis system as OH1 for FC dipole calculation
    # # note: OH2 should not change as a function of OH because it is fixed in the scan (i.e. rotX2/Y2 array should be the same value repeated)
    # rotX = (OHVecs[:, 0] * cosT) + (OHVecs[:, 2] * sinT)  # Keep in case, we need coordinates again!
    # rotZ = (OHVecs[:, 2] * cosT) - (OHVecs[:, 0] * sinT)
    # rotY = OHVecs[:, 1]
    # rotX2 = (OH2Vecs[:, 0] * cosT) + (OH2Vecs[:, 2] * sinT)
    # rotZ2 = (OH2Vecs[:, 2] * cosT) - (OH2Vecs[:, 0] * sinT)
    # rotY2 = OH2Vecs[:, 1]
    # rotXoo = (OOVecs[:, 0] * cosT) + (OOVecs[:, 2] * sinT)
    # rotZoo = (OOVecs[:, 2] * cosT) - (OOVecs[:, 0] * sinT)
    # rotYoo = OOVecs[:, 1]
    # rotXa1 = (OHa1Vecs[:, 0] * cosT) + (OHa1Vecs[:, 2] * sinT)
    # rotZa1 = (OHa1Vecs[:, 2] * cosT) - (OHa1Vecs[:, 0] * sinT)
    # rotYa1 = OHa1Vecs[:, 1]
    # rotXa2 = (OHa2Vecs[:, 0] * cosT) + (OHa2Vecs[:, 2] * sinT)
    # rotZa2 = (OHa2Vecs[:, 2] * cosT) - (OHa2Vecs[:, 0] * sinT)
    # rotYa2 = OHa2Vecs[:, 1]
    # # note: the Z axis does not change, but the components still matter!!
    if ComptoPlot == "Mag":
        comp = np.sqrt(rotMuZ ** 2 + rotMuX ** 2)
        dip = np.sqrt(dipX ** 2 + dipY ** 2 + dipZ ** 2)
    else:
        if ComptoPlot == "X":
            comp = rotMuX
            dip = dipX
        elif ComptoPlot == "Z":
            comp = rotMuZ
            dip = dipZ
        else:
            raise Exception(f"ComptoPlot {ComptoPlot} unrecognized...")
    return comp, dip, np.array((cosT, sinT))  # need to pull out cos/sin rotations to apply to deriv

def pull_HOHpltDat(dataDict, DipDerivs, xy_ranges, water_idx, ComptoPlot):
    """Pulls the data needed to make OH vs Delta MU plots, keyed by HOH values"""
    sort_idx = np.lexsort((dataDict["xyData"][:, 0], dataDict["xyData"][:, 1]))  # sort by HOH (1) then OH (0)
    resortYX = dataDict["xyData"][sort_idx]  # resort so OH is fast and HOH is slow
    eq_idx = np.argmin(dataDict["Energies"][sort_idx])  # eq_idx based on YX sort
    eq_coords = resortYX[eq_idx]  # [OH, HOH] atomic units
    grid_len = len(np.unique(dataDict["xyData"][:, 0]))  # use the length of the unique x-values to reshape grid
    squareYX_full = resortYX.reshape(grid_len, grid_len, 2)  # num of HOH x num of OH
    DipYX = dataDict["RotatedDipoles"][sort_idx]  # x, y, z rotated dipoles in the yx order
    squareDip_full = DipYX.reshape(grid_len, grid_len, 3)  # num of HOH x num of OH x XYZ comps
    sCoords = dataDict["RotatedCoords"][sort_idx]
    sqCoords = sCoords.reshape(grid_len, grid_len, sCoords.shape[1], 3)

    # cut down the Y values (HOH) here to only plot within 20% of wfn maximum - decreases # of lines plotted
    cut_Y = np.where((xy_ranges[1, 0] < squareYX_full[:, 0, 1]) & (squareYX_full[:, 0, 1] < xy_ranges[1, 1]))[0]
    squareYX = squareYX_full[cut_Y, :, :]
    squareDip = squareDip_full[cut_Y, :, :]
    squareCoords = sqCoords[cut_Y, :, :, :]

    hoh_idx = np.argwhere(squareYX[:, 0, 1] == eq_coords[1]).squeeze()  # index in the square arrays where the HOH is @ equilibrium
    oh_idx = np.argwhere(squareYX[0, :, 0] == eq_coords[0]).squeeze()  # index in the square arrays where the OH is @ equilibrium
    if water_idx[0] == 0:  # monomer
        # the x & y are nonzero in the monomer
        ## we calculate the MONOMER fc using the original dipole coordinates!! This is ok, because the magnitude can
        ## not change based of the embedding.
        FC = abs(squareDip[hoh_idx, oh_idx, 1] / (2 * eq_coords[0] * np.cos(eq_coords[1] / 2)))
        print("fixed Charge:", FC)
        eqComp, eqDip, eqRotor = monomer_FC(squareYX[hoh_idx], squareCoords[hoh_idx], squareDip[hoh_idx], ComptoPlot, FC)
        eqComp = eqComp[oh_idx]
        eqDip = eqDip[oh_idx]
        eqRotor = eqRotor[:, oh_idx]
    else:  # dimer
        # the x & z are nonzero in the dimer
        eqCoords = squareCoords[hoh_idx, oh_idx, :, :]  # (6, 3)
        qD = calc_qDonor(eqCoords, squareDip[hoh_idx, oh_idx, :])
        qA = calc_qAcceptor(eqCoords, squareDip[hoh_idx, oh_idx, :], qD)
        print("qA : ", qA, "qD : ", qD)
        eqComp, eqDip, eqRotor = dimer_FC(squareYX[hoh_idx], squareCoords[hoh_idx], squareDip[hoh_idx],
                                          water_idx, ComptoPlot, qD, qA)
        eqComp = eqComp[oh_idx]
        eqDip = eqDip[oh_idx]
        eqRotor = eqRotor[:, oh_idx]

    # SET COLOR VALUES
    cmap1 = plt.get_cmap("Blues_r")  # set all colors for map
    counter1 = 0
    max1 = len(np.argwhere(squareYX[:, 0, 1] < eq_coords[1]))  # the maximum number of HOHs plotted UNDER eq
    cmap2 = plt.get_cmap("Reds")
    counter2 = 1
    max2 = len(np.argwhere(squareYX[:, 0, 1] > eq_coords[1]))  # the maximum number of HOHs plotted OVER eq
    # CALCULATE EQ VALUES - will match ComptoPlot
    PltDat = dict()
    for idx in np.arange(len(squareYX)):  # pull data for one HOH value
        yx = squareYX[idx]  # OH, HOH where OH is fast HOH is slow
        HOH = yx[0, 1]
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
        # OhIdx = np.argwhere(yx[:, 0] == eq_coords[0]).squeeze()
        if water_idx[0] == 0:
            comp, dip, rotor = monomer_FC(yx, squareCoords[idx], squareDip[idx], ComptoPlot, FC)
        else:
            comp, dip, rotor = dimer_FC(yx, squareCoords[idx], squareDip[idx], water_idx, ComptoPlot, qD, qA)
        HOHdeg = int(np.rint(HOH * (180 / np.pi)))  # convert value for legend
        # print(f"For angle {HOHdeg}: FC - {eqComp}, Dip - {eqDip}")
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
        # pull the dipole derivatives to plot as the "linear dipole"
        if ComptoPlot == "Mag":
            norm = np.linalg.norm((DipDerivs["x"]["firstOH"], DipDerivs["y"]["firstOH"], DipDerivs["z"]["firstOH"]))
            y_deriv = norm * x_eq0
            coefsDeriv = [0, 0, 0, norm]
        else:
            if water_idx[0] == 0:  # monomer in X/Y plane
                if ComptoPlot == "X":
                    deriv = (DipDerivs["x"]["firstOH"] * eqRotor[0]) + (DipDerivs["y"]["firstOH"] * eqRotor[1])
                    y_deriv =  deriv * x_eq0
                    coefsDeriv = [0, 0, 0, deriv]
                elif ComptoPlot == "Y":
                    deriv = (DipDerivs["y"]["firstOH"] * eqRotor[0]) - (DipDerivs["x"]["firstOH"] * eqRotor[1])
                    y_deriv = deriv * x_eq0
                    coefsDeriv = [0, 0, 0, deriv]
            else:  # dimer in X/Z plane
                if ComptoPlot == "X":
                    deriv = (DipDerivs["x"]["firstOH"] * eqRotor[0]) + (DipDerivs["z"]["firstOH"] * eqRotor[1])
                    y_deriv =  deriv * x_eq0
                    coefsDeriv = [0, 0, 0, deriv]
                elif ComptoPlot == "Z":
                    deriv = (DipDerivs["z"]["firstOH"] * eqRotor[0]) - (DipDerivs["x"]["firstOH"] * eqRotor[1])
                    y_deriv = deriv * x_eq0
                    coefsDeriv = [0, 0, 0, deriv]
        degDat = {"x_SI": x_ang,  # OH values to be plotted on X-axis
                  "x_eq0": x_eq0,  # OH values shifted so eq is 0 (for expansion)
                  # "FC": FC,  # the calculated Fixed Charge for this HOH
                  "y_charges": y_charges,  # FC to be plotted on Y-axis
                  "y_dips": y_dips,  # Dipole moments to be plotted on Y-axis/ If "Mag" then Magnitude rotated to OH
                  "y_deriv": y_deriv,  # Dipole Derivatives based on FD
                  "DerivCoeffs": coefsDeriv,  # the slope for the dipole derivative
                  "FcCoeffs": coefsFC,  # polyfit coefs of FC line
                  "DipCoeffs": coefsDip,  # polyfit coefs of Dipole line
                  "MFC": MFC}  # color to be used for the marker
        PltDat[HOHdeg] = degDat
    return PltDat

## deleted pull_OHpltDat function.. can pull back from archive, will need to rework.. FC (at least and probably more) is not accurate.

def plot_FCDipvsOH(fig_label, dataDict, DipDerivs, xy_ranges, water_idx, ComptoPlot=None, EQonly=False, Xaxis="OH"):
    plt.rcParams.update({'font.size': 24, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(12, 8), dpi=216)
    PltDat = pull_HOHpltDat(dataDict, DipDerivs, xy_ranges, water_idx, ComptoPlot)
    # vals = [88, 96, 104, 112, 120]
    vals = [88, 104, 120]
    Xlabel = r"$r_{\mathrm{OH}} (\mathrm{\AA})$"
    # if Xaxis == "OH":
    #     PltDat = pull_HOHpltDat(dataDict, DipDerivs, xy_ranges, water_idx, ComptoPlot)
    #     vals = [88, 96, 104, 112, 120]
    #     Xlabel = r"$r_{\mathrm{OH}} (\mathrm{\AA})$"
    # elif Xaxis == "HOH":
    #     PltDat = pull_OHpltDat(dataDict, xy_ranges, water_idx, ComptoPlot)
    #     if water_idx[0] == 0:
    #         vals = [0.911, 0.961, 1.011]
    #     elif water_idx[1] == 4:
    #         vals = [0.91, 0.96, 1.01]
    #     elif water_idx[1] == 5:
    #         vals = [0.918, 0.968, 1.018]
    #     else:
    #         raise Exception(f"Can not determine OH vals for water idx {water_idx}")
    #     Xlabel = r"$\theta_{\mathrm{HOH}} (^{\circ})$"
    # # a dictionary keyed by HOH vals holding all data needed to make the plots
    # else:
    #     raise Exception(f"Can not determine data for X-axis {Xaxis}")
    if EQonly:
        eqDict = PltDat[vals[1]]
        # plot FC points
        plt.plot(eqDict["x_SI"], eqDict["y_charges"], marker="s", color="k", markerfacecolor="forestgreen",
                 markersize=10, markeredgewidth=1, linestyle="None", label="Fixed Charge Model")
        f = np.poly1d(eqDict["FcCoeffs"])
        plt.plot(eqDict["x_SI"], f(eqDict["x_eq0"]), "--", color="k")
        # plot Dipole points
        plt.plot(eqDict["x_SI"], eqDict["y_dips"], marker="o", color="k", markerfacecolor="rebeccapurple",
                 markersize=10, markeredgewidth=1, linestyle="None", label="Full Dipole")
        f1 = np.poly1d(eqDict["DipCoeffs"])
        plt.plot(eqDict["x_SI"], f1(eqDict["x_eq0"]), "-", color="k")
        # plot linear dipole
        plt.plot(eqDict["x_SI"], eqDict["y_deriv"], "-", color="fuchsia", label="Linear Dipole", zorder=10,
                 linewidth=3.0)
    else:
        for deg in PltDat:  # pull data for one HOH value...
            if deg in vals:
                degDict = PltDat[deg]
                # include the plot of the Linear Dipole as well
                if deg == vals[1]:  # only plot @ equilibrium so it only plots once
                    plt.plot(degDict["x_SI"], degDict["y_deriv"], "-", color="fuchsia", #label="Linear Dipole",
                             zorder=8, linewidth=3.5)
                    # plot the FC
                    plt.plot(degDict["x_SI"], degDict["y_charges"], marker="s", color="k", markerfacecolor=degDict["MFC"],
                             markersize=10, markeredgewidth=1, linestyle="None", zorder=15,
                             label=np.round(PltDat[deg]["FcCoeffs"][3], 8))
                    f = np.poly1d(degDict["FcCoeffs"])
                    plt.plot(degDict["x_SI"], f(degDict["x_eq0"]), "--", color="k", zorder=12)
                    # plot the rest of the Dipole plots
                    plt.plot(degDict["x_SI"], degDict["y_dips"], marker="o", color="k", markerfacecolor=degDict["MFC"],
                             markersize=10, markeredgewidth=1, linestyle="None", zorder=16,
                             label=np.round(PltDat[deg]["DipCoeffs"][3], 8))
                    f1 = np.poly1d(degDict["DipCoeffs"])
                    plt.plot(degDict["x_SI"], f1(degDict["x_eq0"]), "-", color="k", zorder=13)
                else:
                    # plot the FC
                    plt.plot(degDict["x_SI"], degDict["y_charges"], marker="s", color="k", markerfacecolor=degDict["MFC"],
                             markersize=10, markeredgewidth=1, linestyle="None",
                             label=np.round(PltDat[deg]["FcCoeffs"][3], 8))
                    f = np.poly1d(degDict["FcCoeffs"])
                    plt.plot(degDict["x_SI"], f(degDict["x_eq0"]), "--", color=degDict["MFC"])
                    # plot the rest of the Dipole plots
                    plt.plot(degDict["x_SI"], degDict["y_dips"], marker="o", color="k", markerfacecolor=degDict["MFC"],
                             markersize=10, markeredgewidth=1, linestyle="None",
                             label=np.round(PltDat[deg]["DipCoeffs"][3], 8))
                    f1 = np.poly1d(degDict["DipCoeffs"])
                    plt.plot(degDict["x_SI"], f1(degDict["x_eq0"]), "-", color=degDict["MFC"])


    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel(Xlabel)
    plt.ylabel(r"$\Delta \mu_{%s}$" % ComptoPlot)
    plt.ylim(-0.40, 0.40)
    plt.yticks(np.arange(-0.4, 0.6, step=0.2))
    plt.tight_layout()
    if EQonly:
        figname = fig_label + "DeltaMu_" + ComptoPlot + "_" + Xaxis + "DipFCplot_EQonlyDeriv_42423.png"
    else:
        figname = fig_label + "DeltaMu_" + ComptoPlot + "_" + Xaxis + "DipFCplot_3vals.png"
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

def plotFCDipSlopes(fig_label, dataDict, DipDerivs, xy_ranges, water_idx, ComptoPlot=None):
    plt.rcParams.update({'font.size': 20, 'legend.fontsize': 18})
    fig = plt.figure(figsize=(8, 8), dpi=216)
    PltDat = pull_HOHpltDat(dataDict, DipDerivs, xy_ranges, water_idx, ComptoPlot)
    x_dat = []
    FCslopes = []
    Dipslopes = []
    for deg in PltDat:
        x_dat.append(deg)
        FCslopes.append(PltDat[deg]["FcCoeffs"][3])
        Dipslopes.append(PltDat[deg]["DipCoeffs"][3])
        plt.plot(deg, PltDat[deg]["FcCoeffs"][3], marker="s", color='k', markerfacecolor="forestgreen",
                 markeredgewidth=1, markersize=10)
        if deg == 104:
            plt.plot(deg, PltDat[deg]["DipCoeffs"][3], marker="o", color='k', markerfacecolor=PltDat[deg]["MFC"],
                     markeredgewidth=1, markersize=10)
        else:
            plt.plot(deg, PltDat[deg]["DipCoeffs"][3], marker="o", color='k', markerfacecolor="rebeccapurple",
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
    plt.plot(x_dat, f(x_shift), "--", color='forestgreen', label=np.round(coefs[3], 8), zorder=-1)
    plt.plot(x_dat, f1(x_shift), "-", color='rebeccapurple', label=np.round(coefs1[3], 8), zorder=-2)
    if ComptoPlot == "X":
        plt.ylim(0.0, 1.0)  # X-axis
    elif ComptoPlot == "Y" or ComptoPlot == "Z":
        plt.ylim(-0.5, 0.5)  # Y/Z-axis
    # plt.ylabel(r"Slope of $\Delta \mu_{%s}$" % ComptoPlot)
    # plt.xlabel(r"$\theta_{\mathrm{HOH}} (^\circ)$")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(fig_label, dpi=fig.dpi, bbox_inches="tight")
