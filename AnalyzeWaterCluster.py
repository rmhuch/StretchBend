import os
import numpy as np
from Converter import Constants
from GmatrixElements import GmatrixStretchBend

class AnalyzeOneWaterCluster:
    def __init__(self, ClusterObj):
        self.ClusterObj = ClusterObj
        self._Gphiphi = None  # G-matrix element of bend at equilibrium (calc'ed using GmatrixStretchBend)
        self._Grr = None  # G-matrix element of stretch at equilibrium (calc'ed using GmatrixStretchBend)
        self._FDFrequency = None  # uses 2nd deriv of five point FD and g-matrix to calculate frequency (wavenumbers)
        self._FDIntensity = None  # uses 1st deriv of five point FD and g-matrix to calculate bend intensity (?)
        self._StretchDipoleDerivs = None  # returns SB derivative for both OHs
        self._StretchFrequency = None
        self._SBGmatrix = None
        self._StretchBendIntensity = None
        self._ResultsDict = None

    @property
    def Gphiphi(self):
        if self._Gphiphi is None:
            # masses are hard coded so that this plays nice with monomer (H, O, H) and tetramer (O, H, H)
            self._Gphiphi = GmatrixStretchBend.calc_Gphiphi(m1=Constants.mass("H", to_AU=True),
                                                            m2=Constants.mass("O", to_AU=True),
                                                            m3=Constants.mass("H", to_AU=True),
                                                            r12=self.ClusterObj.waterIntCoords["R12"],
                                                            r23=self.ClusterObj.waterIntCoords["R23"],
                                                            phi123=self.ClusterObj.waterIntCoords["HOH"])
        return self._Gphiphi

    @property
    def Grr(self):
        if self._Grr is None:
            self._Grr = GmatrixStretchBend.calc_Grr(m1=Constants.mass("O", to_AU=True),
                                                    m2=Constants.mass("H", to_AU=True))
        return self._Grr

    @property
    def FDFrequency(self):
        if self._FDFrequency is None:
            self._FDFrequency = self.calc_FDFrequency()
        return self._FDFrequency

    @property
    def FDIntensity(self):
        if self._FDIntensity is None:
            self._FDIntensity = self.calc_FDIntensity()
        return self._FDIntensity

    @property
    def StretchDipoleDerivs(self):
        if self._StretchDipoleDerivs is None:
            self._StretchDipoleDerivs = self.calc_StretchDipoleDerivs()
        return self._StretchDipoleDerivs

    @property
    def StretchFrequency(self):
        if self._StretchFrequency is None:
            self._StretchFrequency = self.calc_StretchFrequency()
        return self._StretchFrequency

    @property
    def SBGmatrix(self):
        if self._SBGmatrix is None:
            self._SBGmatrix = self.calcSBgmatrix()
        return self._SBGmatrix

    @property
    def StretchBendIntensity(self):
        if self._StretchBendIntensity is None:
            self._StretchBendIntensity = self.calc_StretchBendIntensity()
        return self._StretchBendIntensity

    def calc_PotentialDerivatives(self):
        from McUtils.Zachary import finite_difference
        HOH = self.ClusterObj.FDBdat["HOH Angles"]
        ens = self.ClusterObj.FDBdat["Energies"]
        # print(Constants.convert(ens - min(ens), "wavenumbers", to_AU=False))
        deriv = finite_difference(HOH, ens, 2, stencil=len(HOH), only_center=True)
        # derivative in hartree/radian^2
        return deriv

    def calc_NMFrequency(self):
        # pull F internals (get from special formchk command) and G and use matmul and eig to calc eigenvalues
        NMfreqs = ...
        return NMfreqs

    def calc_FDFrequency(self):
        # calc derivative of potential
        deriv = self.calc_PotentialDerivatives()
        # print(deriv)
        # use g-matrix to calculate the frequency
        freq = np.sqrt(deriv * self.Gphiphi)
        # return Constants.convert(freq, "wavenumbers", to_AU=False)
        return freq

    def calc_FDIntensity(self):
        from McUtils.Zachary import finite_difference
        derivs = np.zeros(3)
        comp_intents = np.zeros(3)
        for i, val in enumerate(["X", "Y", "Z"]):
            derivs[i] = finite_difference(self.ClusterObj.FDBdat["HOH Angles"], self.ClusterObj.FDBdat["RotDipoles"][:, i],
                                          1, stencil=len(self.ClusterObj.FDBdat["HOH Angles"]), only_center=True)
            #  electron charge * bohr /radians
            Gphiphi_wave = Constants.convert(self.Gphiphi, "wavenumbers", to_AU=False)
            comp_intents[i] = (abs(derivs[i])**2 / (0.393456 ** 2)) * ((1/2) * Gphiphi_wave) * 2.506
        intensity = np.sum(comp_intents)
        return intensity

    def calc_StretchDipoleDerivs(self):
        from McUtils.Zachary import finite_difference
        # O H H ordered
        dip_deriv = self.ClusterObj.FDBdat["RotDipoleDerivatives"]  # in A.U.
        # pull carts for each atom in Water O, H1, H2
        carts = self.ClusterObj.FDBdat["RotCartesians"]
        R1plus = self.ClusterObj.FDBdat["RotR1plus"]
        R1minus = self.ClusterObj.FDBdat["RotR1minus"]
        R2plus = self.ClusterObj.FDBdat["RotR2plus"]
        R2minus = self.ClusterObj.FDBdat["RotR2minus"]
        H1_deriv = np.zeros((5, 3))  # this will store d mu_(x, y, z) / d roh in each FD step
        H2_deriv = np.zeros((5, 3))
        for n, step in enumerate(carts):
            dXdR1 = (R1plus[n] - R1minus[n]) / (2 * self.ClusterObj.FDBdat["delta"])
            dXdR2 = (R2plus[n] - R2minus[n]) / (2 * self.ClusterObj.FDBdat["delta"])
            H1_deriv[n, :] = np.tensordot(dXdR1, dip_deriv[n], axes=[[0, 1], [0, 1]])
            H2_deriv[n, :] = np.tensordot(dXdR2, dip_deriv[n], axes=[[0, 1], [0, 1]])
        SB1deriv = np.zeros(3)
        SB2deriv = np.zeros(3)
        for j, comp in enumerate(["X", "Y", "Z"]):  # store SB derivative as components
            SB1deriv[j] = finite_difference(self.ClusterObj.FDBdat["HOH Angles"], H1_deriv[:, j], 1,
                                            stencil=len(self.ClusterObj.FDBdat["HOH Angles"]), only_center=True)
            SB2deriv[j] = finite_difference(self.ClusterObj.FDBdat["HOH Angles"], H2_deriv[:, j], 1,
                                            stencil=len(self.ClusterObj.FDBdat["HOH Angles"]), only_center=True)
        self.H1_deriv = H1_deriv
        self.H2_deriv = H2_deriv
        return SB1deriv, SB2deriv  # D / Ang * amu^1/2

    def calc_StretchFrequency(self):
        # uses the harmonic displacements to determine which OH stretch frequency corresponds to
        # which OH in the wateridx
        # call two largest freqs and their disps
        freqs = np.flip(self.ClusterObj.HarmFreqs)
        OH_disps = np.flip(self.ClusterObj.HarmDisps[-2:], axis=0)
        OH_freqs = np.zeros(2)  # 1 x number of OH disps (2) array
        for i, mode in enumerate(OH_disps):
            OH_normDisps = np.linalg.norm(mode, axis=1)  # take the norm of the x, y, z displacements for each atom
            sort_idx = np.argsort(OH_normDisps)
            sort_disps = OH_normDisps[sort_idx]
            disps_diff = sort_disps[-1] - sort_disps[-2]
            # print(sort_disps[-2], sort_disps[-1])
            if disps_diff <= 0.2:
                print(f"Harmonic Displacements are only {disps_diff} different, using geometric mean of OH stretches.")
                OH_freqs[i] = np.sqrt(freqs[0] * freqs[1])  # always two highest freqs
            elif sort_idx[-1] == self.ClusterObj.wateridx[1]:  # check if largest displacement is of H1
                OH_freqs[0] = freqs[i]
            elif sort_idx[-1] == self.ClusterObj.wateridx[2]:  # check if largest displacement is of H2
                OH_freqs[1] = freqs[i]
            else:
                raise Exception(f"Atom {sort_idx[-1]} is neither H1 or H2.")
        freqs_AU = Constants.convert(OH_freqs, "wavenumbers", to_AU=True)  # return frequencies in AU
        return freqs_AU

    def calcSBgmatrix(self):
        Grr = GmatrixStretchBend.calc_Grr(m1=Constants.mass("O", to_AU=True), m2=Constants.mass("H", to_AU=True))
        GrrP = GmatrixStretchBend.calc_Grrprime(m1=Constants.mass("O", to_AU=True),
                                                phi123=self.ClusterObj.waterIntCoords["HOH"])
        Grphi = GmatrixStretchBend.calc_Grphi(m2=Constants.mass("O", to_AU=True),
                                              r23=self.ClusterObj.waterIntCoords["R23"],
                                              phi123=self.ClusterObj.waterIntCoords["HOH"])
        # Grphi2 = GmatrixStretchBend.calc_Grphi(m2=Constants.mass("O", to_AU=True),
        #                                        r23=self.ClusterObj.waterIntCoords["R12"],
        #                                        phi123=self.ClusterObj.waterIntCoords["HOH"])
        Gmat = np.array([[Grr, GrrP, Grphi],
                        [GrrP, Grr, Grphi],
                        [Grphi, Grphi, self.Gphiphi]])
        return Gmat

    def calc_StretchIntensity(self):
        # frequencies in au
        intensities = np.zeros(2)
        Grr_wave = Constants.convert(self.Grr, "wavenumbers", to_AU=False)
        for i, deriv in enumerate([self.H1_deriv[2], self.H2_deriv[2]]):
            comps = np.zeros(3)
            for j, val in enumerate(["X", "Y", "Z"]):
                # deriv in AU
                comps[j] = ((abs(deriv[j])**2) / (0.393456 ** 2)) * ((1/2) * Grr_wave) * 2.506
            intensities[i] = np.sum(comps)
        return intensities

    def calc_StretchBendIntensity(self):
        # frequencies in au
        deltaT = (1/2) * self.Gphiphi / self.FDFrequency
        intensities = np.zeros(2)
        for i, deriv in enumerate(self.StretchDipoleDerivs):
            print(Constants.convert(self.StretchFrequency[i], "wavenumbers", to_AU=False))
            deltaR = (1 / 2) * self.Grr / self.StretchFrequency[i]
            freq = deltaT * deltaR * (self.StretchFrequency[i] + self.FDFrequency)
            freq_wave = Constants.convert(freq, "wavenumbers", to_AU=False)
            comps = np.zeros(3)
            for j, val in enumerate(["X", "Y", "Z"]):
                # deriv in AU
                comps[j] = ((abs(deriv[j])**2) / (0.393456 ** 2)) * freq_wave * 2.506
            intensities[i] = np.sum(comps)
        return intensities


