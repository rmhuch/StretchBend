from BuildWaterCluster import *
from AnalyzeWaterCluster import AnalyzeOneWaterCluster
from NormalModes import *

# water4 = BuildTetCage(num_waters=4, isotopologue="Hw4", FDBstep="0.5")
# monomer = BuildMonomer()
# dimer = BuildDimer(num_waters=2, isotopologue="Hw2", FDBstep="0.5")
# water31 = BuildTetThreeOne(num_waters=4, isotopologue="Hw2", FDBstep="0.5")
# print(water2.waterIntCoords["HOH"] * (180/np.pi))
water5 = BuildPentRing(num_waters=5, isotopologue="Hw5", FDBstep="0.5")
analyzeObj = AnalyzeOneWaterCluster(ClusterObj=water5)
print(analyzeObj.StretchBendIntensity)
print(analyzeObj.calc_StretchIntensity())
# print("FD: ", analyzeObj.FDFrequency)
# print("FD: ", analyzeObj.FDIntensity)

# ham = mass_weight(monomer.Fchkdat.hessian, monomer.massarray, num_coord=(3*3))
# nm = norms(ham)
# print(Constants.convert(np.sqrt(nm["freq2"][-3:]), "wavenumbers", to_AU=False))
# print(nm["qn"][:, -3:].T.reshape(3, 3, 3))





