import os
import numpy as np
from Converter import Constants

class AnalyzeOneWaterCluster:
    def __init__(self, ClusterObj):
        self.ClusterObj = ClusterObj
        self._Gphiphi = None  # G-matrix element of bend at equilibrium
        self._FDFrequency = None  # uses 2nd deriv of five point FD and g-matrix to calculate frequency (wavenumbers)
        self._FDIntensity = None  # uses 1st deriv of five point FD and g-matrix to calculate bend intensity (?)

    @property
    def Gphiphi(self):
        if self._Gphiphi is None:
            self._Gphiphi = self.calc_Gphiphi(self.ClusterObj.waterIntCoords["R12"],
                                              self.ClusterObj.waterIntCoords["R23"],
                                              self.ClusterObj.waterIntCoords["HOH"])
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

    def calc_Gphiphi(self, r12, r23, phi123):
        m1 = self.ClusterObj.massarray[self.ClusterObj.wateridx[0]]
        m2 = self.ClusterObj.massarray[self.ClusterObj.wateridx[1]]
        m3 = self.ClusterObj.massarray[self.ClusterObj.wateridx[2]]
        term1 = 1 / (m1 * r12**2)
        term2 = 1 / (m3 * r23**2)
        term3 = (1 / m2) * ((1 / r12**2) + (1 / r23**2) - (2 * np.cos(phi123) / (r12 * r23)))
        return term1 + term2 + term3

    def calc_PotentialDerivatives(self):
        from McUtils.Zachary import finite_difference
        HOH = self.ClusterObj.FDBdat["HOH Angles"]
        ens = self.ClusterObj.FDBdat["Energies"]
        # print(Constants.convert(ens-min(ens), "wavenumbers", to_AU=False))
        deriv = finite_difference(HOH, ens, 2, stencil=5, only_center=True)
        # derivative in hartree/radian^2
        return deriv

    def calc_FDFrequency(self):
        # calc derivative of potential
        deriv = self.calc_PotentialDerivatives()
        # use g-matrix to calculate the frequency
        freq = np.sqrt(deriv * self.Gphiphi)
        return Constants.convert(freq, "wavenumbers", to_AU=False)

    def calc_FDIntensity(self):
        from McUtils.Zachary import finite_difference
        derivs = np.zeros(3)
        for i, val in enumerate(["X", "Y", "Z"]):
            derivs[i] = finite_difference(self.ClusterObj.FDBdat["HOH Angles"], self.ClusterObj.FDBdat["Dipoles"][:, i],
                                          1, stencil=5, only_center=True)
            #  dipole atomic units/radians
        derivs_debye = Constants.convert(derivs, "debye", to_AU=False)
        intensity = ...
        return intensity



