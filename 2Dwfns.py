import numpy as np
import os
from PyDVR.DVR import *
from McUtils.Plots import ContourPlot
from Converter import Constants
from GmatrixElements import GmatrixStretchBend

def calc_EQgeom(dataDict):
    min_arg = np.argmin(dataDict["Energies"])
    eqCarts = dataDict["Cartesians"][min_arg]
    print(min_arg)
    vec12 = eqCarts[1] - eqCarts[0]
    vec23 = eqCarts[2] - eqCarts[0]
    r12 = np.linalg.norm(vec12)
    r23 = np.linalg.norm(vec23)
    ang = (np.dot(vec12, vec23)) / (r12 * r23)  # angstroms
    HOH = (np.arccos(ang))  # radians
    return Constants.convert(r12, "angstroms", to_AU=True), Constants.convert(r23, "angstroms", to_AU=True), HOH

def run_2D_DVR(dataDict):
    """Runs 2D DVR over the original 2D potential"""
    dvr_2D = DVR("ColbertMillerND")
    name = str(dataDict["DataName"])
    npz_filename = os.path.join(str(dataDict["MainDir"]), f"{name}_2D_DVR.npz")
    x = Constants.convert(dataDict["xyData"][:, 0], "angstroms", to_AU=True)
    y = (dataDict["xyData"][:, 1]*2) * (np.pi/180)
    xy = np.column_stack((x, y))  # xy in bohr/radians
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    muOH = 1/(1/mO + 1/mH)
    r12, r23, HOH = calc_EQgeom(dataDict)
    muHOH = GmatrixStretchBend.calc_Gphiphi(m1=mH, m2=mO, m3=mH,
                                            r12=r12, r23=r23, phi123=HOH)
    pot = Constants.convert(dataDict["Energies"], "wavenumbers", to_AU=True)
    res = dvr_2D.run(potential_grid=np.column_stack((xy, pot)),
                     divs=(25, 25), mass=[muOH, (1/muHOH)], num_wfns=5,
                     domain=((min(xy[:, 0]),  max(xy[:, 0])), (min(xy[:, 1]), max(xy[:, 1]))),
                     results_class=ResultsInterpreter)
    dvr_grid = Constants.convert(res.grid, "angstroms", to_AU=False)
    dvr_pot = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
    ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
    # res.plot_potential(plot_class=ContourPlot, plot_units="wavenumbers", colorbar=True).show()
    wfns = res.wavefunctions.wavefunctions
    # ResultsInterpreter.wfn_contours(res)
    np.savez(npz_filename, grid=[dvr_grid], potential=[dvr_pot], energy_array=ens, wfns_array=wfns)
    print(f"saved data to f{npz_filename}")
    return npz_filename

if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MainDir = os.path.join(docs, "stretch_bend", "RyanOctomer")
    logs = ["w2_R5B", "w6_R5B", "w6a_R5B"]  #, "w1_RB"]
    for name in logs:
        dat = np.load(os.path.join(MainDir, f"{name}_GaussRes.npz"), allow_pickle=True)
        run_2D_DVR(dat)
