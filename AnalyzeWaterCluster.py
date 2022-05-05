import os
import numpy as np
from Converter import Constants
from GmatrixElements import GmatrixStretchBend

class AnalyzeOneWaterCluster:
    def __init__(self, ClusterObj):
        self.ClusterObj = ClusterObj
        self._Gphiphi = None  # G-matrix element of bend at equilibrium (calc'ed using GmatrixHOH)
        self._FDFrequency = None  # uses 2nd deriv of five point FD and g-matrix to calculate frequency (wavenumbers)
        self._FDIntensity = None  # uses 1st deriv of five point FD and g-matrix to calculate bend intensity (?)
        self._StretchDipoleDerivs = None
        self._StetchFrequency = None
        self._SBGmatrix = None
        self._StretchBendIntensity = None

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
    def StetchFrequency(self):
        if self._StetchFrequency is None:
            self._StetchFrequency = self.calc_StetchFrequency()
        return self._StetchFrequency

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
        print(Constants.convert(ens - min(ens), "wavenumbers", to_AU=False))
        deriv = finite_difference(HOH, ens, 2, stencil=5, only_center=True)
        # derivative in hartree/radian^2
        return deriv

    def calc_NMFrequency(self):
        # pull F internals (get from special formchk command) and G and use matmul and eig to calc eigenvalues
        NMfreqs = ...
        return NMfreqs

    def calc_FDFrequency(self):
        # calc derivative of potential
        deriv = self.calc_PotentialDerivatives()
        print(deriv)
        # use g-matrix to calculate the frequency
        freq = np.sqrt(deriv * self.Gphiphi)
        return Constants.convert(freq, "wavenumbers", to_AU=False)

    def calc_FDIntensity(self):
        from McUtils.Zachary import finite_difference
        derivs = np.zeros(3)
        comp_intents = np.zeros(3)
        for i, val in enumerate(["X", "Y", "Z"]):
            derivs[i] = finite_difference(self.ClusterObj.FDBdat["HOH Angles"], self.ClusterObj.FDBdat["Dipoles"][:, i],
                                          1, stencil=5, only_center=True)
            #  electron charge * bohr /radians
            Gphiphi_wave = Constants.convert(self.Gphiphi, "wavenumbers", to_AU=False)
            comp_intents[i] = (abs(derivs[i])**2 / (0.393456 ** 2)) * ((1/2) * Gphiphi_wave) * 2.506
        intensity = np.sum(comp_intents)
        return intensity

    def calc_StretchDipoleDerivs(self):
        # something about this is not right.. dropping for now and stepping back to H2O and (H2O)2 4/14
        # pull 9 dip derivs for each atom in Water O, H1, H2
        dip_deriv = self.ClusterObj.FDBdat["Dipole Derivatives"][:, self.ClusterObj.wateridx, :]
        # pull carts for each atom in Water O, H1, H2
        Wcarts = self.ClusterObj.FDBdat["Cartesians"][:, self.ClusterObj.wateridx, :]
        H1_deriv = np.dot(dip_deriv[:, 1, :], ((Wcarts[1] - Wcarts[0]) / self.ClusterObj.FDBdat["R12"]))
        return dip_deriv

    def calc_StetchFrequency(self):
        freq = ...
        return freq

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

    def calc_StretchBendIntensity(self):
        intents = ...
        return intents


