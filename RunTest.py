from BuildWaterCluster import *
from AnalyzeWaterCluster import AnalyzeOneWaterCluster
from NormalModes import *
from Figures import *
from Rotator import get_xyz

# water4 = BuildTetCage(num_atoms=12, isotopologue="Hw4", FDBstep="0.5")
# monomer = BuildMonomer()
# dimer = BuildDimer(num_atoms=6, isotopologue="Hw1", FDBstep="0.5")
# water31 = BuildTetThreeOne(num_atoms=12, isotopologue="Hw4", FDBstep="0.5")
# water6 = BuildHexCage(num_atoms=18, isotopologue="allH", FDBstep="0.5")
# water5pls = BuildTetCageTZ(num_atoms=12, isotopologue="Hw4", FDBstep="0.5")
# analyzeObj = AnalyzeOneWaterCluster(ClusterObj=water5pls)
# print("Bend Freq: ", Constants.convert(analyzeObj.FDFrequency, "wavenumbers", to_AU=False))
# print("Bend: ", analyzeObj.FDIntensity)
# print("Stretch: ", analyzeObj.calc_StretchIntensity())
# print("Stretch + Bend: ", analyzeObj.StretchBendIntensity)
# dr1 = np.linalg.norm(analyzeObj.SBDipoleDerivs[0]) / np.linalg.norm(analyzeObj.StretchDipoleDerivs[0])
# dr2 = np.linalg.norm(analyzeObj.SBDipoleDerivs[1]) / np.linalg.norm(analyzeObj.StretchDipoleDerivs[1])
# print("DerivRatio 1: ", dr1)
# print("DerivRatio 2: ", dr2)
se = Plots(DataSet="DZvsTZ")
# se.make_OH_Sticks()
# se .make_SB_sticks()
se.plotSIvsSBI()
se.plotSBfreqvsSBI()
se.plotSfreqvsSI()
se.plotBfreqvsBI()
se.plotOOvsSBI()
se.plotOHOvsSBI()
se.plotOOvsBI()
se.plotOOvsSI()
se.plotOHOvsBI()
se.plotOHOvsBI()
se.plotOHOvsSI()
se.plotVPTvsSBI()
se.plotVPTvsSBI_AVG()
se.plotDerivRatiovsSBI()




