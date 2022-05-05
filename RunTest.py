from BuildWaterCluster import *
from AnalyzeWaterCluster import AnalyzeOneWaterCluster
from NormalModes import *

# water4 = BuildTetCage(isotopologue="Hw2", FDBstep="1")
# monomer = BuildMonomer()
# dimer = BuildDimer(isotopologue="Hw1", FDBstep="0.5")
# water2 = BuildTetThreeOne(isotopologue="Hw4", FDBstep="0.5")
# print(water2.waterIntCoords["HOH"] * (180/np.pi))
water5 = BuildPentCage(isotopologue="Hw5", FDBstep="1")
analyzeObj = AnalyzeOneWaterCluster(ClusterObj=water5)
# a = analyzeObj.StretchDipoleDerivs
print("FD: ", analyzeObj.FDFrequency)
print("FD: ", analyzeObj.FDIntensity)

# ham = mass_weight(monomer.Fchkdat.hessian, monomer.massarray, num_coord=(3*3))
# nm = norms(ham)
# print(Constants.convert(np.sqrt(nm["freq2"][-3:]), "wavenumbers", to_AU=False))
# print(nm["qn"][:, -3:].T.reshape(3, 3, 3))





