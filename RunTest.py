from BuildWaterCluster import *
from AnalyzeWaterCluster import AnalyzeOneWaterCluster

# water1 = BuildTetCage(isotopologue="Hw1", FDBstep="0.5")
# monomer = BuildMonomer()
dimer = BuildDimer(isotopologue="Hw2", FDBstep="0.5")
analyzeObj = AnalyzeOneWaterCluster(ClusterObj=dimer)
# a = analyzeObj.StretchDipoleDerivs
print("FD: ", analyzeObj.FDFrequency)
print("FD: ", analyzeObj.FDIntensity)





