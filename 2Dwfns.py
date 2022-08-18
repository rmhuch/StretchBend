import numpy as np
import os
from PyDVR.DVR import *
from McUtils.Plots import ContourPlot
import matplotlib.pyplot as plt
from Converter import Constants
from GmatrixElements import GmatrixStretchBend

def calc_EQgeom(dataDict):
    min_arg = np.argmin(dataDict["Energies"])
    eqCarts = dataDict["Cartesians"][min_arg]
    vec12 = eqCarts[1] - eqCarts[0]
    vec23 = eqCarts[2] - eqCarts[0]
    r12 = np.linalg.norm(vec12)
    r23 = np.linalg.norm(vec23)
    ang = (np.dot(vec12, vec23)) / (r12 * r23)  # angstroms
    HOH = (np.arccos(ang))  # radians
    return r12, r23, HOH


def potlint(x=None, y=None):
    from scipy import interpolate
    tck = interpolate.splrep(x, y, s=0)

    def pf(grid, extrap=tck):
        y_fit = interpolate.splev(grid, extrap, der=0)
        return y_fit
    return pf

def run_BendDVR(dataDict):
    """Runs 1D DVR over Bend potential at OH Equilibrium"""
    dvr_1d = DVR("ColbertMiller1D")
    Y = np.unique(dataDict["xyData"][:, 1])*2
    Yrads = Y * (np.pi/180)
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    r12, r23, HOH = calc_EQgeom(dataDict)
    inds = np.argwhere(dataDict["xyData"][:, 0] == np.round(Constants.convert(r12, "angstroms", to_AU=False), 4))
    muHOH = GmatrixStretchBend.calc_Gphiphi(m1=mH, m2=mO, m3=mH,
                                            r12=r12, r23=r23, phi123=HOH)
    muOH = 1/(1/mO + 1/mH)
    pot = np.squeeze(Constants.convert(dataDict["Energies"][inds], "wavenumbers", to_AU=True))
    res = dvr_1d.run(potential_function=potlint(Yrads, pot), divs=100, mass=1/muHOH,
                     domain=(min(Yrads), max(Yrads)), num_wfns=5)
    ens = res.wavefunctions.energies
    print(Constants.convert(ens, "wavenumbers", to_AU=False))
    potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
    grid = res.grid * (180 / np.pi)  # grid in degrees
    plt.plot(grid, potential)
    plt.show()
    return res

def run_StretchDVR(dataDict):
    """Runs 1D DVR over Bend potential at OH Equilibrium"""
    dvr_1d = DVR("ColbertMiller1D")
    X = np.unique(dataDict["xyData"][:, 0])
    Xbohr = Constants.convert(X, "angstroms", to_AU=True)
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    r12, r23, HOH = calc_EQgeom(dataDict)
    inds = np.argwhere(dataDict["xyData"][:, 1] == np.round((HOH * (180 / np.pi) / 2), 4))
    muOH = 1/(1/mO + 1/mH)
    pot = np.squeeze(Constants.convert(dataDict["Energies"][inds], "wavenumbers", to_AU=True))
    res = dvr_1d.run(potential_function=potlint(Xbohr, pot), divs=100, mass=muOH,
                     domain=(min(Xbohr)-Constants.convert(0.15, "angstroms", to_AU=True), max(Xbohr)), num_wfns=5)
    ens = res.wavefunctions.energies
    print(Constants.convert(ens, "wavenumbers", to_AU=False))
    potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
    grid = Constants.convert(res.grid, "angstroms", to_AU=False)
    plt.plot(grid, potential)
    plt.show()
    return res

def make_one_Wfn_plot(ResDict, wfnn, idx=(0, 1)):
    # complete analysis and gather data
    x = Constants.convert(ResDict.grid, "angstroms", to_AU=False)
    wfn0g = wfnn[:, idx[0]] * 100
    wfn1g = wfnn[:, idx[1]] * 100
    # plot
    colors = ["b", "r", "g", "indigo", "teal", "mediumvioletred"]
    plt.plot(x, np.repeat(0, len(x)), "-k", linewidth=3)
    plt.plot(x, wfn0g, color=colors[idx[0]], linewidth=3)
    plt.plot(x, wfn1g, color=colors[idx[1]], linewidth=3)
    plt.show()

def run_2DDVR(dataDict):
    """Runs 2D DVR over the original 2D potential"""
    dvr_2D = DVR("ColbertMillerND")
    fname = str(dataDict["DataName"])
    npz_filename = os.path.join(str(dataDict["MainDir"]), f"{fname}_2D_DVRPA.npz")
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    muOH = 1/(1/mO + 1/mH)
    r12, r23, HOH = calc_EQgeom(dataDict)
    muHOH = GmatrixStretchBend.calc_Gphiphi(m1=mH, m2=mO, m3=mH,
                                            r12=r12, r23=r23, phi123=HOH)
    res = dvr_2D.run(potential_grid=np.column_stack((dataDict["xyData"], dataDict["Energies"])),
                     divs=(31, 31), mass=[muOH, (1/muHOH)], num_wfns=6,
                     domain=((min(dataDict["xyData"][:, 0]),  max(dataDict["xyData"][:, 0])),
                             (min(dataDict["xyData"][:, 1]), max(dataDict["xyData"][:, 1]))),
                     results_class=ResultsInterpreter)
    dvr_grid = res.grid
    dvr_pot = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
    ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
    print(ens - ens[0])
    # res.plot_potential(plot_class=ContourPlot, plot_units="wavenumbers", colorbar=True).show()
    wfns = res.wavefunctions.wavefunctions
    ResultsInterpreter.wfn_contours(res)
    np.savez(npz_filename, grid=[dvr_grid], potential=[dvr_pot], energy_array=ens, wfns_array=wfns)
    print(f"saved data to f{npz_filename}")
    return npz_filename

if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MainDir = os.path.join(docs, "stretch_bend", "RyanOctomer")
    # logs = ["w2_R5B", "w6_R5B", "w6a_R5B"]
    # for name in logs:
    name = "w1_RB"
    dat = np.load(os.path.join(MainDir, f"{name}_bigDataDictPA.npz"), allow_pickle=True)
    a = run_2DDVR(dat)
    # make_one_Wfn_plot(a, a.wavefunctions.wavefunctions)
