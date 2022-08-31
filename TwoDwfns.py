from McUtils.Plots import ContourPlot
from PyDVR.DVR import *
from Converter import Constants
from DVRtools import *
from GmatrixElements import GmatrixStretchBend

def run_2DDVR(dataDict, water_idx, print_ens=False, plot_potential=False, plot_wfns=False):
    """Runs 2D DVR over the original 2D potential, called from 'AnalyzeIntensityClusters.py' """
    dvr_2D = DVR("ColbertMillerND")
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    muOH = 1/(1/mO + 1/mH)
    r12, r23, HOH = calc_EQgeom(dataDict, water_idx=water_idx)
    muHOH = GmatrixStretchBend.calc_Gphiphi(m1=mH, m2=mO, m3=mH,
                                            r12=r12, r23=r23, phi123=HOH)
    res = dvr_2D.run(potential_grid=np.column_stack((dataDict["xyData"], dataDict["Energies"])),
                     divs=(31, 31), mass=[muOH, (1/muHOH)], num_wfns=6,
                     domain=((min(dataDict["xyData"][:, 0]),  max(dataDict["xyData"][:, 0])),
                             (min(dataDict["xyData"][:, 1]), max(dataDict["xyData"][:, 1]))),
                     results_class=ResultsInterpreter)
    if print_ens:
        ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        print(ens - ens[0])
    if plot_potential:
        res.plot_potential(plot_class=ContourPlot, plot_units="wavenumbers", colorbar=True).show()
    if plot_wfns:
        ResultsInterpreter.wfn_contours(res)
    ResDict = dict(grid=[res.grid], potential=[res.potential_energy.diagonal()],
                   energy_array=res.wavefunctions.energies, wfns_array=res.wavefunctions.wavefunctions)
    return ResDict

