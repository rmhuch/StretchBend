import numpy as np
import os
from Converter import Constants

class AnalyzeIntensityCluster:
    def __init__(self, ClusterObj, HChargetoPlot=None, TDMtype=None):
        self.ClusterObj = ClusterObj
        if HChargetoPlot is None:
            self.Htag = self.ClusterObj.WaterIdx[1]
        else:
            self.Htag = HChargetoPlot
        if TDMtype is None:
            self.TDMtype = "Dipole Surface"
        else:
            self.TDMtype = TDMtype
        self.SysStr = self.ClusterObj.SysStr
        self._BendDVRData = None
        self._StretchDVRData = None
        self._SBDVRData = None
        self._wfns = None
        self._SBwfnRanges = None
        self._DipDerivs = None
        self._tdms = None
        self._twoDintensities = None

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

    @property
    def twoDintensities(self):
        if self._twoDintensities is None:
            self._twoDintensities = self.getting2DIntense()
        return self._twoDintensities

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
        # OHvsQ = plot_DeltaQvsHOH(fig_label, self.ClusterObj.BigScanDataDict, self.SBwfnRanges, self.ClusterObj.WaterIdx,
        #                          HchargetoPlot=self.Htag)
        HOHvsQ = plot_DeltaQvsOH(fig_label, self.ClusterObj.BigScanDataDict, self.SBwfnRanges, self.ClusterObj.WaterIdx,
                                 HchargetoPlot=self.Htag)
        # slabel = os.path.join(self.ClusterObj.MainFigDir, f"{self.ClusterObj.SysStr}H{self.Htag}_QslopevOH.png")
        if self.ClusterObj.num_atoms == 1:
            HB = None
        elif self.ClusterObj.num_atoms == 2:
            HB = self.ClusterObj.Hbound
        else:
            raise Exception("Can not determine if scan is H-bound or not")
        # plotChargeSlopes(slabel, OHvsQ, xlabel="OH", Hbound=HB, HChargetoPlot=self.Htag)
        s2label = os.path.join(self.ClusterObj.MainFigDir, f"{self.ClusterObj.SysStr}H{self.Htag}_QslopevHOH4.png")
        plotChargeSlopes(s2label, HOHvsQ, xlabel="HOH", Hbound=HB, HChargetoPlot=self.Htag)

    def make_FixedChargePlots(self):
        from ChargePlots import plot_FixedCharge, plotFCSlopes
        fig_label = os.path.join(self.ClusterObj.MainFigDir, self.ClusterObj.SysStr)
        if self.ClusterObj.num_atoms == 1:
            HB = None
        elif self.ClusterObj.num_atoms == 2:
            HB = self.ClusterObj.Hbound
        else:
            raise Exception("Can not determine if scan is H-bound or not")
        for i in ["X", "Y", "Z"]:
            FCdat = plot_FixedCharge(fig_label, self.ClusterObj.BigScanDataDict, self.SBwfnRanges,
                                     self.ClusterObj.WaterIdx, ComptoPlot=i)
            slabel = os.path.join(self.ClusterObj.MainFigDir, f"{self.ClusterObj.SysStr}H{self.Htag}_{i}FCslopevHOH.png")
            plotFCSlopes(slabel, FCdat, Hbound=HB, ComptoPlot=i)
        # i = "Mag"
        # FCdat = plot_FixedCharge(fig_label, self.ClusterObj.BigScanDataDict, self.SBwfnRanges,
        #                          self.ClusterObj.WaterIdx, ComptoPlot=i)
        # slabel = os.path.join(self.ClusterObj.MainFigDir, f"{self.ClusterObj.SysStr}H{self.Htag}_{i}FCslopevHOH.png")
        # plotFCSlopes(slabel, FCdat, Hbound=HB, ComptoPlot=i)

    def make_DipolePlots(self):
        from DipolePlots import plot_DipolevsOH, plotDipSlopes
        fig_label = os.path.join(self.ClusterObj.MainFigDir, self.ClusterObj.SysStr)
        # for i in ["X", "Y", "Z"]:
        #     dipVSoh = plot_DipolevsOH(fig_label, self.ClusterObj.BigScanDataDict, self.SBwfnRanges, DipoletoPlot=i)
        #     slabel = os.path.join(self.ClusterObj.MainFigDir, f"{self.ClusterObj.SysStr}H{self.Htag}_{i}DslopevHOH.png")
        #     plotDipSlopes(slabel, dipVSoh, DipoletoPlot=i)
        i = "Mag"
        dipVSoh = plot_DipolevsOH(fig_label, self.ClusterObj.BigScanDataDict, self.SBwfnRanges, DipoletoPlot=i)
        slabel = os.path.join(self.ClusterObj.MainFigDir, f"{self.ClusterObj.SysStr}H{self.Htag}_{i}DslopevHOH.png")
        plotDipSlopes(slabel, dipVSoh, DipoletoPlot=i)

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
            freq = self.SBDVRData["energy_array"][i] - self.SBDVRData["energy_array"][0]
            print(freq)
            for j in np.arange(3):
                matEl[j] = np.dot(self.wfns[:, 0], (trans_mom[:, j] * self.wfns[:, i]))
                comp_intents[j] = (matEl[j]) ** 2
            intensities[i-1] = np.sum(comp_intents) * freq * 2.506 / (0.393456 ** 2)
            print(intensities[i-1])
        return intensities
