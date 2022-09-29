import numpy as np
import os
from Converter import Constants

class AnalyzeIntensityCluster:
    def __init__(self, ClusterObj, HChargetoPlot=None):
        self.ClusterObj = ClusterObj
        if HChargetoPlot is None:
            self.Htag = self.ClusterObj.WaterIdx[1]
        else:
            self.Htag = HChargetoPlot
        self.SysStr = self.ClusterObj.SysStr
        self._BendDVRData = None
        self._StretchDVRData = None
        self._SBDVRData = None
        self._wfns = None
        self._SBwfnRanges = None
        self._DipDerivs = None
        self._tdms = None

    @property
    def BendDVRData(self):
        if self._BendDVRData is None:
            from OneDwfns import run_BendDVR
            bendDVRfile = os.path.join(self.ClusterObj.ClusterDir, f"{self.SysStr}BendDVR.npz")
            if os.path.exists(bendDVRfile):
                self._BendDVRData = np.load(bendDVRfile, allow_pickle=True)
            else:
                bendDat = run_BendDVR(self.ClusterObj.BigScanDataDict, self.ClusterObj.WaterIdx,
                                      print_ens=True, plot_potential=False, plot_wfns=True)
                np.savez(bendDVRfile, **bendDat)
                print(f"saved data to {bendDVRfile}")
                self._BendDVRData = bendDat
        return self._BendDVRData

    @property
    def StretchDVRData(self):
        if self._StretchDVRData is None:
            from OneDwfns import run_StretchDVR
            stretchDVRfile = os.path.join(self.ClusterObj.ClusterDir, f"{self.SysStr}StretchDVR.npz")
            if os.path.exists(stretchDVRfile):
                self._StretchDVRData = np.load(stretchDVRfile, allow_pickle=True)
            else:
                stretchDat = run_StretchDVR(self.ClusterObj.BigScanDataDict, self.ClusterObj.WaterIdx,
                                            print_ens=True, plot_potential=False, plot_wfns=True)
                np.savez(stretchDVRfile, **stretchDat)
                print(f"saved data to {stretchDVRfile}")
                # All data saved in atomic units
                self._StretchDVRData = stretchDat
        return self._StretchDVRData

    @property
    def SBDVRData(self):
        if self._SBDVRData is None:
            from TwoDwfns import run_2DDVR
            SBDVRfile = os.path.join(self.ClusterObj.ClusterDir, f"{self.SysStr}2D_DVR.npz")
            if os.path.exists(SBDVRfile):
                self._SBDVRData = np.load(SBDVRfile, allow_pickle=True)
            else:
                SBdat = run_2DDVR(self.ClusterObj.BigScanDataDict, self.ClusterObj.WaterIdx,
                                  print_ens=False, plot_potential=False, plot_wfns=True)
                np.savez(SBDVRfile, **SBdat)
                print(f"saved data to {SBDVRfile}")
        return self._SBDVRData

    @property
    def wfns(self):
        if self._wfns is None:
            self._wfns = self.SBDVRData["wfns_array"]
        return self._wfns

    @property
    def SBwfnRanges(self):
        if self._SBwfnRanges is None:
            self._SBwfnRanges = self.find_WfnRanges()
        return self._SBwfnRanges

    @property
    def DipDerivs(self):
        if self._DipDerivs is None:
            fname = os.path.join(self.ClusterObj.MainDir, "w1", f"{self.SysStr}DipDerivs.npz")
            if os.path.exists(fname):
                derivs = np.load(fname, allow_pickle=True)
                self._DipDerivs = {k: derivs[k].item() for k in ["x", "y", "z"]}
                self._DipDerivs["eqDipole"] = derivs["eqDipole"]
            else:
                self._DipDerivs = self.calc_DipDerivs()
        return self._DipDerivs

    @property
    def tdms(self):
        if self._tdms is None:
            self._tdms = self.calc_all2Dmus()
        return self._tdms

    def find_WfnRanges(self):
        xy_ranges = np.zeros((2, 2))
        for i, res in enumerate([self.StretchDVRData, self.BendDVRData]):
            GS = res["wfns_array"][:, 0]
            MaxIdx = np.argmax(GS)
            Two0Percent = GS[MaxIdx] * 0.2
            LminIdx = np.argmin(np.abs(GS[:MaxIdx] - Two0Percent))
            RminIdx = np.argmin(np.abs(GS[MaxIdx:] - Two0Percent))
            xy_ranges[i] = [res["grid"][LminIdx], res["grid"][MaxIdx + RminIdx]]
        return xy_ranges

    def make_ChargePlots(self):
        from ChargePlots import plot_DeltaQvsHOH, plot_DeltaQvsOH, plotChargeSlopes
        fig_label = os.path.join(self.ClusterObj.MainFigDir, self.ClusterObj.SysStr)
        OHvsQ = plot_DeltaQvsHOH(fig_label, self.ClusterObj.BigScanDataDict, self.SBwfnRanges, self.ClusterObj.WaterIdx,
                                 HchargetoPlot=self.Htag)
        HOHvsQ = plot_DeltaQvsOH(fig_label, self.ClusterObj.BigScanDataDict, self.SBwfnRanges, self.ClusterObj.WaterIdx,
                                 HchargetoPlot=self.Htag)
        slabel = os.path.join(self.ClusterObj.MainFigDir, f"{self.ClusterObj.SysStr}H{self.Htag}_QslopevOH.png")
        if self.ClusterObj.num_atoms == 1:
            HB = None
        elif self.ClusterObj.num_atoms == 2:
            HB = self.ClusterObj.Hbound
        else:
            raise Exception("Can not determine if scan is H-bound or not")
        plotChargeSlopes(slabel, OHvsQ, xlabel="OH", Hbound=HB, HChargetoPlot=self.Htag)
        s2label = os.path.join(self.ClusterObj.MainFigDir, f"{self.ClusterObj.SysStr}H{self.Htag}_QslopevHOH.png")
        plotChargeSlopes(s2label, HOHvsQ, xlabel="HOH", Hbound=HB, HChargetoPlot=self.Htag)

    def calc_DipDerivs(self):
        from SurfaceDerivatives import calc_allDerivs
        derivDict = calc_allDerivs(self.ClusterObj.SmallScanDataDict)
        fname = os.path.join(self.ClusterObj.ClusterDir, f"{self.ClusterObj.SysStr}DipDerivs.npz")
        np.savez(fname, **derivDict)
        print(f"Data saved to {fname}")
        return derivDict

    def calc_DerivNorms(self):
        norms = dict()
        for key in self.DipDerivs["x"]:
            norms[key] = np.linalg.norm((self.DipDerivs["x"][key], self.DipDerivs["y"][key], self.DipDerivs["z"][key]))
        return norms

    def calc_all2Dmus(self):
        from functools import reduce
        from operator import mul
        from TDMexpansions import TM2Dexpansion
        bigGrid = self.SBDVRData["grid"][0]
        npts = reduce(mul, bigGrid.shape[:-1], 1)
        Grid = np.reshape(bigGrid, (npts, bigGrid.shape[-1]))
        params = dict()
        fd_ohs = self.ClusterObj.SmallScanDataDict["ROH"]
        fd_hohs = self.ClusterObj.SmallScanDataDict["HOH"]
        params["delta_oh"] = np.round(Grid[:, 0] - fd_ohs[2], 3)
        params["delta_hoh"] = np.round(Grid[:, 1] - fd_hohs[2], 3)
        twodeedms = dict()
        twodeedms["dipSurf"] = self.ClusterObj.BigScanDataDict["RotatedDipoles"]
        twodeedms["cubic"] = TM2Dexpansion.cubic_DM(params, self.DipDerivs)
        twodeedms["quad"] = TM2Dexpansion.quad_DM(params, self.DipDerivs)
        twodeedms["quadOH"] = TM2Dexpansion.quadOH_DM(params, self.DipDerivs)
        twodeedms["quadbilin"] = TM2Dexpansion.quadBILIN_DM(params, self.DipDerivs)
        twodeedms["lin"] = TM2Dexpansion.lin_DM(params, self.DipDerivs)
        return twodeedms

