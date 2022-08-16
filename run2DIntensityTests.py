import os
import numpy as np
from Converter import Constants

class TwoDHarmonicWfnDipoleExpansions:
    def __init__(self, sys_str, TDMtype):
        self.sys_str = sys_str
        self.TDMtype = TDMtype
        self._MainDir = None
        self._FDdat = None
        self._SurfaceScanRes = None
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
            dat = np.load(os.path.join(self.MainDir, f"{self.sys_str}_smallDataDict.npy"), allow_pickle=True)
            self._FDdat = dat.item()
        return self._FDdat

    @property
    def SurfaceScanRes(self):
        if self._SurfaceScanRes is None:
            if self.sys_str == "w1":
                self._SurfaceScanRes = np.load(os.path.join(self.MainDir, f"{self.sys_str}_RB_GaussRes.npz"),
                                               allow_pickle=True)
            else:
                self._SurfaceScanRes = np.load(os.path.join(self.MainDir, f"{self.sys_str}_R5B_GaussRes.npz"),
                                               allow_pickle=True)
        return self._SurfaceScanRes

    @property
    def DVRdat(self):
        if self._DVRdat is None:
            # energies in wavenumbers grid in bohr/radian
            if self.sys_str == "w1":
                self._DVRdat = np.load(os.path.join(self.MainDir, f"{self.sys_str}_RB_2D_DVR_bigGrid.npz"),
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
    def dipSurf(self):
        if self._dipSurf is None:
            if self.sys_str == "w1":
                self._dipSurf = np.load(os.path.join(self.MainDir, f"{self.sys_str}_RB_bigrotdips_OHO.npy"),
                                               allow_pickle=True)
            else:
                self._dipSurf = np.load(os.path.join(self.MainDir, f"{self.sys_str}_R5B_rotdips_OHO.npy"),
                                               allow_pickle=True)
        return self._dipSurf

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
        # Grid = self.DVRdat["grid"].reshape((961, 2))  # grid for whole scan / in bohr/radians
        params = dict()
        derivs = np.load(os.path.join(self.MainDir, f"{self.sys_str}_DipCoefs.npz"), allow_pickle=True)
        newDerivs = {k: derivs[k].item() for k in ["x", "y", "z"]}
        params["eqDipole"] = derivs["eqDip"]  # place in EQ Dipole from the small scan
        fd_ohs = self.FDdat["ROH"]
        fd_hohs = self.FDdat["HOH"]
        params["delta_oh"] = Grid[:, 0] - fd_ohs[2]
        params["delta_hoh"] = Grid[:, 1] - fd_hohs[2]
        twodeedms = dict()
        twodeedms["dipSurf"] = self.dipSurf
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
        intensities = np.zeros(len(self.wfns[0, :]) - 1)
        matEl = np.zeros(3)
        comp_intents = np.zeros(3)
        for i in np.arange(1, len(self.wfns[0, :])):  # starts at 1 to only loop over exciting states
            print("excited state: ", i)
            freq = self.DVRdat["energy_array"][i] - self.DVRdat["energy_array"][0]
            print(freq)
            for j in np.arange(3):
                super_es = trans_mom[:, j] * self.wfns[:, i]
                matEl[j] = np.dot(self.wfns[:, 0], super_es) ** 2
                comp_intents[j] = matEl[j] * freq * 2.506 / (0.393456 ** 2)
            intensities[i-1] = np.sum(comp_intents)
        return intensities

