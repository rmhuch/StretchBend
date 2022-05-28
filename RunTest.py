from BuildWaterCluster import *
from AnalyzeWaterCluster import AnalyzeOneWaterCluster
from NormalModes import *
from Figures import *

# water4 = BuildTetCage(num_waters=4, isotopologue="Hw1", FDBstep="0.5")
# monomer = BuildMonomer()
# dimer = BuildDimer(num_waters=2, isotopologue="Hw1", FDBstep="0.5")
# water31 = BuildTetThreeOne(num_waters=4, isotopologue="Hw1", FDBstep="0.5")
# print(water2.waterIntCoords["HOH"] * (180/np.pi))
# water5 = BuildHexCage(num_waters=6, isotopologue="Hw6", FDBstep="0.5")
# print(water5.HarmFreqs)
# analyzeObj = AnalyzeOneWaterCluster(ClusterObj=water5)
# print("Stretch + Bend: ", analyzeObj.StretchBendIntensity)
# print("Stretch: ", analyzeObj.calc_StretchIntensity())
# print("FD: ", analyzeObj.FDFrequency)
# print("Bend: ", analyzeObj.FDIntensity)
se = Plots()
se.plotOHOvsSBI()
se.plotOOvsSBI()





