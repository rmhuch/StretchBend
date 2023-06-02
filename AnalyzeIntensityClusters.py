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
        self._HarmTwoDIntensities = None

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
            fname = os.path.join(self.ClusterObj.ClusterDir, f"{self.SysStr}DipDerivs.npz")
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

    @property
    def HarmTwoDIntensities(self):
        if self._HarmTwoDIntensities is None:
            self._HarmTwoDIntensities = self.gettingHarm2DIntense()
        return self._HarmTwoDIntensities

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

    def calcInternals(self, carts):
        """For a given set of cartesian coordinates (FOR ONE GEOMETRY), calculate the two OH bonds and HOH angle."""
        vec12 = carts[self.ClusterObj.WaterIdx[1]] - carts[self.ClusterObj.WaterIdx[0]]
        vec23 = carts[self.ClusterObj.WaterIdx[2]] - carts[self.ClusterObj.WaterIdx[0]]
        r12 = np.linalg.norm(vec12)
        r23 = np.linalg.norm(vec23)
        ang = (np.dot(vec12, vec23)) / (np.linalg.norm(vec12) * np.linalg.norm(vec23))
        HOH = np.arccos(ang)
        return HOH, r12, r23

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
            slabel = os.path.join(self.ClusterObj.MainFigDir,
                                  f"{self.ClusterObj.SysStr}H{self.Htag}_{i}FCslopevHOH.png")
            plotFCSlopes(slabel, FCdat, Hbound=HB, ComptoPlot=i)
        # i = "Mag"
        # FCdat = plot_FixedCharge(fig_label, self.ClusterObj.BigScanDataDict, self.SBwfnRanges,
        #                          self.ClusterObj.WaterIdx, ComptoPlot=i)
        # slabel = os.path.join(self.ClusterObj.MainFigDir, f"{self.ClusterObj.SysStr}H{self.Htag}_{i}FCslopevHOH.png")
        # plotFCSlopes(slabel, FCdat, Hbound=HB, ComptoPlot=i)

    def make_NCPlots(self):
        from NaturalCharges import plot_NCs
        fig_label = os.path.join(self.ClusterObj.MainFigDir, self.ClusterObj.SysStr)
        eq_idx = np.argmin(self.ClusterObj.BigScanDataDict["Energies"])
        eq_coords = self.ClusterObj.BigScanDataDict["xyData"][eq_idx]
        plot_NCs(fig_label, self.ClusterObj.BigScanDataDict, eq_coords, self.ClusterObj.WaterIdx, self.SBwfnRanges)

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

    def make_FCDipCompPlots(self, EQonly=False, Inds=["Mag", "X"]):
        from DipFCPlots import plotFCDipSlopes, plot_FCDipvsOH, plotFCvsHOH
        fig_label = os.path.join(self.ClusterObj.MainFigDir, self.ClusterObj.SysStr)
        for i in Inds:
            plot_FCDipvsOH(fig_label, self.ClusterObj.BigScanDataDict, self.DipDerivs, self.SBwfnRanges,
                           self.ClusterObj.WaterIdx, ComptoPlot=i, EQonly=EQonly, Xaxis="OH")
            if EQonly is False:
                slabel = os.path.join(self.ClusterObj.MainFigDir,
                                      f"{self.ClusterObj.SysStr}{i}_DipFCslope_onecolor.png")
                plotFCDipSlopes(slabel, self.ClusterObj.BigScanDataDict, self.DipDerivs, self.SBwfnRanges,
                                self.ClusterObj.WaterIdx, ComptoPlot=i)

    def calc_DipDerivs(self):
        from SurfaceDerivatives import calc_allDerivs
        derivDict = calc_allDerivs(self.ClusterObj.SmallScanDataDict)
        fname = os.path.join(self.ClusterObj.ClusterDir, f"{self.ClusterObj.SysStr}DipDerivs.npz")
        np.savez(fname, **derivDict)
        print(f"Data saved to {fname}")
        return derivDict

    def calc_PotDerivs(self):
        from SurfaceDerivatives import calc_PotDerivs
        derivDict = calc_PotDerivs(self.ClusterObj.SmallScanDataDict)
        fname = os.path.join(self.ClusterObj.ClusterDir, f"{self.ClusterObj.SysStr}PotDerivs.npz")
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
        # np.savetxt("XXXDVRgrid.txt", Grid) <-- data files compiled to send Anne for her DVR
        # np.savetxt("XXXDVRpot.txt", self.SBDVRData["potential"].T)
        params = dict()
        fd_ohs = self.ClusterObj.SmallScanDataDict["ROH"]
        fd_hohs = self.ClusterObj.SmallScanDataDict["HOH"]
        params["delta_oh"] = np.round(Grid[:, 0] - fd_ohs[2], 3)
        params["delta_hoh"] = np.round(Grid[:, 1] - fd_hohs[2], 3)
        twodeedms = dict()
        twodeedms["dipSurf"] = self.ClusterObj.BigScanDataDict["RotatedDipoles"]
        # np.savetxt("XXXDVRdipoleX.txt", self.ClusterObj.BigScanDataDict["RotatedDipoles"][:, 0])
        # np.savetxt("XXXDVRdipoleZ.txt", self.ClusterObj.BigScanDataDict["RotatedDipoles"][:, 2])
        twodeedms["quartic"] = TM2Dexpansion.quartic_DM(params, self.DipDerivs)
        twodeedms["cubic"] = TM2Dexpansion.cubic_DM(params, self.DipDerivs)
        twodeedms["quad"] = TM2Dexpansion.quad_DM(params, self.DipDerivs)
        twodeedms["quad_only"] = TM2Dexpansion.quad_only_DM(params, self.DipDerivs)
        twodeedms["quaddiag"] = TM2Dexpansion.quadDIAG_DM(params, self.DipDerivs)
        twodeedms["quaddiag_only"] = TM2Dexpansion.quadDIAG_only_DM(params, self.DipDerivs)
        twodeedms["quadOH"] = TM2Dexpansion.quadOH_DM(params, self.DipDerivs)
        twodeedms["quadHOH"] = TM2Dexpansion.quadHOH_DM(params, self.DipDerivs)
        twodeedms["quadbilin"] = TM2Dexpansion.quadBILIN_DM(params, self.DipDerivs)
        twodeedms["quadbilin_only"] = TM2Dexpansion.quadBILIN_only_DM(params, self.DipDerivs)
        twodeedms["lin"] = TM2Dexpansion.lin_DM(params, self.DipDerivs)
        twodeedms["linOH"] = TM2Dexpansion.linOH_DM(params, self.DipDerivs)
        twodeedms["linHOH"] = TM2Dexpansion.linHOH_DM(params, self.DipDerivs)
        return twodeedms

    def getting2DIntense(self):
        if self.TDMtype == "Dipole Surface":
            trans_mom = self.tdms["dipSurf"]
        elif self.TDMtype == "Quartic":
            trans_mom = self.tdms["quartic"]
        elif self.TDMtype == "Cubic":
            trans_mom = self.tdms["cubic"]
        elif self.TDMtype == "Quadratic":
            trans_mom = self.tdms["quad"]
        elif self.TDMtype == "Quadratic Only":
            trans_mom = self.tdms["quad_only"]
        elif self.TDMtype == "Quadratic OH Only":  # contains only second derivative of mu wrt OH
            trans_mom = self.tdms["quadOH"]
        elif self.TDMtype == "Quadratic HOH Only":  # contains only second derivative of mu wrt HOH
            trans_mom = self.tdms["quadHOH"]
        elif self.TDMtype == "Quadratic Diagonal":
            trans_mom = self.tdms["quaddiag"]
        elif self.TDMtype == "Quadratic Diagonal Only":  # contains only second derivatives of mu wrt OH & HOH
            trans_mom = self.tdms["quaddiag_only"]
        elif self.TDMtype == "Quadratic Bilinear":
            trans_mom = self.tdms["quadbilin"]
        elif self.TDMtype == "Quadratic Bilinear Only":  # contains only second derivative of mu wrt OH/HOH (mixed term)
            trans_mom = self.tdms["quadbilin_only"]
        elif self.TDMtype == "Linear":
            trans_mom = self.tdms["lin"]
        elif self.TDMtype == "Linear OH Only":  # contains only first derivative of mu wrt OH
            trans_mom = self.tdms["linOH"]
        elif self.TDMtype == "Linear HOH Only":  # contains only first derivative of mu wrt HOH
            trans_mom = self.tdms["linHOH"]
        else:
            raise Exception("Can't determine TDM type.")
        # use identified transition moment to calculate the intensities
        print(self.TDMtype)
        intensities = np.zeros(len(self.wfns[0, :]) - 1)
        matEl = np.zeros(3)
        matEl_D = np.zeros(3)
        comp_intents = np.zeros(3)
        # HOH, 2 HOH, OH, 3 HOH, SB
        # freq = [1572.701, 3117.353, 3744.751, 4690.054, 5294.391]
        for i in np.arange(1, len(self.wfns[0, :])):  # starts at 1 to only loop over exciting states
            freq = self.SBDVRData["energy_array"][i] - self.SBDVRData["energy_array"][0]
            freq_wave = Constants.convert(freq, "wavenumbers", to_AU=False)
            for j in np.arange(3):
                matEl[j] = np.dot(self.wfns[:, 0], (trans_mom[:, j] * self.wfns[:, i]))
                matEl_D[j] = matEl[j] / 0.393456
                comp_intents[j] = (abs(matEl[j]) ** 2) * freq_wave * 2.506 / (0.393456 ** 2)
            intensities[i - 1] = np.sum(comp_intents)
            if i == 5:
                print("excited state: ", i)
                print(freq_wave)
                print("Component Mat El (D):", matEl_D)
                print(intensities[i - 1])
        return intensities

    def gettingHarm2DIntense(self):
        from GmatrixElements import GmatrixStretchBend
        """Complete GF Analysis to calculate frequencies and intensities in Harmonic Normal Mode Frame based on the 
        2D potential energy surface from gaussian. """
        eqIdx = np.argmin(self.ClusterObj.SmallScanDataDict["Energies"])
        eq_geom = self.ClusterObj.SmallScanDataDict["RotatedCoords"][eqIdx]
        # calculate g-matrix elements at the equilibrium geometry
        HOH, r12, r23 = self.calcInternals(eq_geom)
        mO = Constants.mass("O", to_AU=True)
        mH = Constants.mass("H", to_AU=True)
        Gphiphi = GmatrixStretchBend.calc_Gphiphi(m1=mH, m2=mO, m3=mH,
                                                  r12=r12, r23=r23, phi123=HOH)
        Grr = GmatrixStretchBend.calc_Grr(m1=mH, m2=mO)
        Grphi = GmatrixStretchBend.calc_Grphi(m2=mO, r23=r23, phi123=HOH)
        # pull/calculate potential surface derivs (Force Constants)
        PotDerivs = self.calc_PotDerivs()
        Frr = PotDerivs["secondOH"]
        Fphiphi = PotDerivs["secondHOH"]
        Frphi = PotDerivs["mixedHOH_OH"]
        # complete GF analysis to get out frequencies and transformation matrices
        G = np.array(((Gphiphi, Grphi), (Grphi, Grr)))
        F = np.array(((Fphiphi, Frphi), (Frphi, Frr)))
        ham = np.matmul(G, F)
        freq2, L = np.linalg.eigh(ham)
        # Frequencies in LOCAL mode
        # OHfreq = np.sqrt(PotDerivs["secondOH"] / muOH)
        # HOHfreq = np.sqrt(PotDerivs["secondHOH"] / (1/Gphiphi))
        OHfreq = np.sqrt(freq2[1])
        HOHfreq = np.sqrt(freq2[0])
        print("SB freq:", Constants.convert((OHfreq + HOHfreq), "wavenumbers", to_AU=False))
        # put it all together
        dT = np.sqrt((0.5 * Gphiphi) / HOHfreq)
        dr = np.sqrt((0.5 * Grr) / OHfreq)
        S_wave = Constants.convert(OHfreq, "wavenumbers", to_AU=False)
        B_wave = Constants.convert(HOHfreq, "wavenumbers", to_AU=False)
        SB_wave = Constants.convert((OHfreq + HOHfreq), "wavenumbers", to_AU=False)
        matElS = np.zeros(3)
        matElB = np.zeros(3)
        matElSB = np.zeros(3)
        compsS = np.zeros(3)
        compsB = np.zeros(3)
        compsSB = np.zeros(3)
        for j, val in enumerate(["x", "y", "z"]):
            # transform dipole into normal coords (only use for SB)
            mu = np.array(((self.DipDerivs[val]["secondHOH"], self.DipDerivs[val]["mixedHOH_OH"]),
                           (self.DipDerivs[val]["mixedHOH_OH"], self.DipDerivs[val]["secondOH"])))
            muQ = np.matmul(L.T,  (np.matmul(mu, L)))
            matElS[j] =  self.DipDerivs[val]["firstOH"] * dr
            matElB[j] = self.DipDerivs[val]["firstHOH"] * dT
            matElSB[j] =  muQ[0, 1] * dT * dr
            compsS[j] = (matElS[j] ** 2 / (0.393456 ** 2)) * S_wave * 2.506
            compsB[j] = (matElB[j] ** 2 / (0.393456 ** 2)) * B_wave * 2.506
            compsSB[j] = (matElSB[j] ** 2 / (0.393456 ** 2)) * SB_wave * 2.506
        intensityS = np.sum(compsS)
        intensityB = np.sum(compsB)
        intensitySB = np.sum(compsSB)
        print("S intensity: ", intensityS)
        print("S comps: ", matElS)
        print("SB intensity: ", intensitySB)
        print("SB comps: ", matElSB)
        return intensityS, intensityB, intensitySB
