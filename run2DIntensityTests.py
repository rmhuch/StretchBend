import os
import numpy as np
from Converter import Constants

## I THINK THIS IS OLD CODE.. AND NOT CALLED ANYMORE.
## REWRITTEN IN ANALYZEINTENSITYCLUSTERS.PY
class TwoDHarmonicWfnDipoleExpansions:
    def __init__(self, sys_str, TDMtype):
        self.sys_str = sys_str
        self.TDMtype = TDMtype
        self._MainDir = None
        self._FDdat = None
        self._Surfacedat = None
        self._DVRdat = None
        self._wfns = None
        self._dipSurf = None
        self._tdms = None

    @property
    def MainDir(self):
        if self._MainDir is None:
            docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self._MainDir = os.path.join(docs, "stretch_bend", "RyanOctomer")
        return self._MainDir

    @property
    def FDdat(self):
        if self._FDdat is None:
            self._FDdat = np.load(os.path.join(self.MainDir, "w1", "w1_RBdata",
                                               f"{self.sys_str}_smallDataDict.npz"), allow_pickle=True)
        return self._FDdat

    @property
    def Surfacedat(self):
        if self._Surfacedat is None:
            if self.sys_str == "w1":
                self._Surfacedat = np.load(os.path.join(self.MainDir, "w1", "w1_RBdata",
                                                        f"{self.sys_str}_RB_bigDataDict.npz"), allow_pickle=True)
            else:
                self._Surfacedat = np.load(os.path.join(self.MainDir, f"{self.sys_str}_R5B_bigDataDict.npz"),
                                               allow_pickle=True)
            # Full surface Gaussian Output: AtomStr, xyData, Energies, ROTATED cartesians and dipoles
            # (from SurfacePlots.py - log) <- converted to a.u.
        return self._Surfacedat

    @property
    def DVRdat(self):
        if self._DVRdat is None:
            # energies in wavenumbers grid in bohr/radian (from TwoDwfns.py)
            if self.sys_str == "w1":
                self._DVRdat = np.load(os.path.join(self.MainDir, "w1", f"{self.sys_str}_2D_DVR.npz"),
                                               allow_pickle=True)
            else:
                self._DVRdat = np.load(os.path.join(self.MainDir, f"{self.sys_str}_R5B_2D_DVR.npz"),
                                               allow_pickle=True)
        return self._DVRdat

    @property
    def wfns(self):
        if self._wfns is None:
            self._wfns = self.DVRdat["wfns_array"]
        return self._wfns

    @property
    def tdms(self):
        if self._tdms is None:
            self._tdms = self.calc_all2Dmus()
        return self._tdms

    def calc_all2Dmus(self):
        from functools import reduce
        from operator import mul
        from TDMexpansions import TM2Dexpansion
        bigGrid = self.DVRdat["grid"][0]
        npts = reduce(mul, bigGrid.shape[:-1], 1)
        Grid = np.reshape(bigGrid, (npts, bigGrid.shape[-1]))
        params = dict()
        derivs = np.load(os.path.join(self.MainDir, "w1", f"{self.sys_str}_DipCoefs.npz"), allow_pickle=True)
        newDerivs = {k: derivs[k].item() for k in ["x", "y", "z"]}
        params["eqDipole"] = derivs["eqDip"]  # place in EQ Dipole from the small scan
        fd_ohs = self.FDdat["ROH"]
        fd_hohs = self.FDdat["HOH"]
        params["delta_oh"] = np.round(Grid[:, 0] - fd_ohs[2], 3)
        params["delta_hoh"] = np.round(Grid[:, 1] - fd_hohs[2], 3)
        twodeedms = dict()
        twodeedms["dipSurf"] = self.Surfacedat["RotatedDipoles"]
        twodeedms["cubic"] = TM2Dexpansion.cubic_DM(params, newDerivs)
        twodeedms["quad"] = TM2Dexpansion.quad_DM(params, newDerivs)
        twodeedms["quadOH"] = TM2Dexpansion.quadOH_DM(params, newDerivs)
        twodeedms["quadbilin"] = TM2Dexpansion.quadBILIN_DM(params, newDerivs)
        twodeedms["lin"] = TM2Dexpansion.lin_DM(params, newDerivs)
        return twodeedms

    def getting2DIntense(self):
        if self.TDMtype == "Dipole Surface":
            trans_mom = self.tdms["dipSurf"]
        elif self.TDMtype == "Cubic":
            trans_mom = self.tdms["cubic"]
        elif self.TDMtype == "Quadratic":
            trans_mom = self.tdms["quad"]
        elif self.TDMtype == "Quadratic OH only":
            trans_mom = self.tdms["quadOH"]
        elif self.TDMtype == "Quadratic Bilinear":
            trans_mom = self.tdms["quadbilin"]
        elif self.TDMtype == "Linear":
            trans_mom = self.tdms["lin"]
        else:
            raise Exception("Can't determine TDM type.")
        # use identified transition moment to calculate the intensities
        print(self.TDMtype)
        intensities = np.zeros(len(self.wfns[0, :]) - 1)
        matEl = np.zeros(3)
        comp_intents = np.zeros(3)
        # HOH, 2 HOH, OH, 3 HOH, SB
        # freq = [1572.701, 3117.353, 3744.751, 4690.054, 5294.391]
        for i in np.arange(1, len(self.wfns[0, :])):  # starts at 1 to only loop over exciting states
            print("excited state: ", i)
            freq = self.DVRdat["energy_array"][i] - self.DVRdat["energy_array"][0]
            print(freq)
            for j in np.arange(3):
                matEl[j] = np.dot(self.wfns[:, 0], (trans_mom[:, j] * self.wfns[:, i]))
                comp_intents[j] = (matEl[j]) ** 2
            intensities[i-1] = np.sum(comp_intents) * freq * 2.506 / (0.393456 ** 2)
            print(intensities[i-1])
        return intensities

