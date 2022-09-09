from BuildWaterCluster import *
from AnalyzeWaterCluster import AnalyzeOneWaterCluster
from NormalModes import *
from Figures import *
from Rotator import get_xyz
from run2DIntensityTests import TwoDHarmonicWfnDipoleExpansions
from BuildIntensityClusters import *
from AnalyzeIntensityClusters import *

# water4 = BuildTetCage(num_atoms=12, isotopologue="Hw4", FDBstep="0.5")
# monomer = BuildMonomer()
# print(np.linalg.norm(monomer.FDBdat["Dipoles"][eq_idx]))
# dimer = BuildDimer(num_atoms=6, isotopologue="Hw1", FDBstep="0.5")
# water31 = BuildTetThreeOne(num_atoms=12, isotopologue="Hw4", FDBstep="0.5")
# water6 = BuildHexCage(num_atoms=18, isotopologue="allH", FDBstep="0.5")
# water5pls = BuildHexT1(num_atoms=19, isotopologue="Hw5", FDBstep="0.5")
# analyzeObj = AnalyzeOneWaterCluster(ClusterObj=dimer)
# a = analyzeObj.FDIntensity
# print("Stretch derivs", analyzeObj.StretchDipoleDerivs)
# print(np.linalg.norm(analyzeObj.StretchDipoleDerivs[0]), np.linalg.norm(analyzeObj.StretchDipoleDerivs[1]))
# print("SB derivs:", analyzeObj.SBDipoleDerivs)
# print(np.linalg.norm(analyzeObj.SBDipoleDerivs[0]), np.linalg.norm(analyzeObj.SBDipoleDerivs[1]))

# print("Bend Freq: ", Constants.convert(analyzeObj.FDFrequency, "wavenumbers", to_AU=False))
# print("Bend: ", analyzeObj.FDIntensity)
# print("Stretch: ", analyzeObj.calc_StretchIntensity())
# print("Stretch + Bend: ", analyzeObj.StretchBendIntensity)
# dr1 = np.linalg.norm(analyzeObj.SBDipoleDerivs[0]) / np.linalg.norm(analyzeObj.StretchDipoleDerivs[0])
# dr2 = np.linalg.norm(analyzeObj.SBDipoleDerivs[1]) / np.linalg.norm(analyzeObj.StretchDipoleDerivs[1])
# print("DerivRatio 1: ", dr1)
# print("DerivRatio 2: ", dr2)
# se = Plots(DataSet="Neutral")
# se.make_OH_Sticks()
# se .make_SB_sticks()
# se.plotSIvsSBI()
# se.plotSBfreqvsSBI()
# se.plotSfreqvsSI()
# se.plotBfreqvsBI()
# se.plotOOvsSBI()
# se.plotOHOvsSBI()
# se.plotOOvsSI()
# se.plotOHOvsSI()
# se.plotVPTvsSBI()
# se.plotVPTvsSBI_AVG()
# se.plotDerivRatiovsSBI()
# se.plotDerivRatiovsVPT()
#
# tdmTypes = ["Dipole Surface", "Cubic", "Quadratic", "Quadratic Bilinear", "Linear"]
# for tdm in tdmTypes:
#     w1 = TwoDHarmonicWfnDipoleExpansions("w1", tdm)
#     w1.getting2DIntense()

w2 = BuildW2(isotopologue="rigid", Hbound=True)
analyzeObj = AnalyzeIntensityCluster(w2, HChargetoPlot=4)
analyzeObj.make_ChargePlots()

# eq_idx = np.argmin(analyzeObj.ClusterObj.SmallScanDataDict["Energies"])
# print(analyzeObj.ClusterObj.SmallScanDataDict["Cartesians"][eq_idx])
# print(analyzeObj.ClusterObj.SmallScanDataDict["Dipoles"][eq_idx])
# print(np.linalg.norm(analyzeObj.ClusterObj.SmallScanDataDict["Dipoles"][eq_idx]))
# eq_idx = np.argmin(analyzeObj.ClusterObj.BigScanDataDict["Energies"])
# print(analyzeObj.ClusterObj.BigScanDataDict["Cartesians"][eq_idx])
# print(analyzeObj.ClusterObj.BigScanDataDict["Dipoles"][eq_idx])
# print(np.linalg.norm(analyzeObj.ClusterObj.BigScanDataDict["Dipoles"][eq_idx]))
