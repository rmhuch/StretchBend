from PyDVR.DVR import *
import matplotlib.pyplot as plt
from Converter import Constants
from GmatrixElements import GmatrixStretchBend
from DVRtools import *

def run_BendDVR(dataDict, water_idx, print_ens=False, plot_potential=False, plot_wfns=False):
    """Runs 1D DVR over Bend potential at OH Equilibrium"""
    dvr_1d = DVR("ColbertMiller1D")
    Yrads = np.unique(dataDict["xyData"][:, 1])
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    r12, r23, HOH = calc_EQgeom(dataDict, water_idx)
    inds = np.argwhere(np.round(dataDict["xyData"][:, 0], 5) == np.round(r12, 5))
    muHOH = GmatrixStretchBend.calc_Gphiphi(m1=mH, m2=mO, m3=mH, r12=r12, r23=r23, phi123=HOH)
    pot = np.squeeze(dataDict["Energies"][inds])
    res = dvr_1d.run(potential_function=potlint(Yrads, pot), divs=100, mass=1/muHOH,
                     domain=(min(Yrads), max(Yrads)), num_wfns=5)
    ens = res.wavefunctions.energies
    wfns = res.wavefunctions.wavefunctions
    if print_ens:
        print(Constants.convert(ens, "wavenumbers", to_AU=False))
    if plot_potential:
        potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
        plt.plot(res.grid * (180 / np.pi), potential)
        plt.show()
    if plot_wfns:
        make_one_Wfn_plot(res.grid * (180 / np.pi), wfns, idx=(0, 1))
        # idx is the states you want to plot, currently supports up to 6
    ResDict = dict(grid=res.grid, potential=res.potential_energy.diagonal(), energy_array=ens, wfns_array=wfns)
    return ResDict

def run_StretchDVR(dataDict, water_idx, print_ens=False, plot_potential=False, plot_wfns=False):
    """Runs 1D DVR over Bend potential at OH Equilibrium"""
    dvr_1d = DVR("ColbertMiller1D")
    Xbohr = np.unique(dataDict["xyData"][:, 0])
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    r12, r23, HOH = calc_EQgeom(dataDict, water_idx)
    inds = np.argwhere(np.round(dataDict["xyData"][:, 1], 4) == np.round(HOH, 4))
    muOH = 1/(1/mO + 1/mH)
    pot = np.squeeze(dataDict["Energies"][inds])
    res = dvr_1d.run(potential_function=potlint(Xbohr, pot), divs=100, mass=muOH,
                     domain=(min(Xbohr), max(Xbohr)), num_wfns=5)
    ens = res.wavefunctions.energies
    wfns = res.wavefunctions.wavefunctions
    if print_ens:
        print(Constants.convert(ens, "wavenumbers", to_AU=False))
    if plot_potential:
        potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
        plt.plot(Constants.convert(res.grid, "angstroms", to_AU=False), potential)
        plt.show()
    if plot_wfns:
        make_one_Wfn_plot(Constants.convert(res.grid, "angstroms", to_AU=False), wfns, idx=(0, 1))
        # idx is the states you want to plot, currently supports up to 6
    ResDict = dict(grid=res.grid, potential=res.potential_energy.diagonal(), energy_array=ens, wfns_array=wfns)
    return ResDict
