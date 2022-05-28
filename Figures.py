import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from NMParser import pull_VPTblock

class PlotSpectrum:
    def __init__(self, ClusterObj, filetag):
        self.ClusterObj = ClusterObj
        self.filetag = filetag
        self.SaveDir = self.ClusterObj.ClusterDir
        self._VPTdat = None

    @property
    def VPTdat(self):
        if self._VPTdat is None:
            self._VPTdat = pull_VPTblock(self.ClusterObj.eqlog)
        return self._VPTdat

    def make_OH_Sticks(self):
        plt.rcParams.update({'font.size': 20})
        fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
        markerline, stemline, baseline = plt.stem(self.VPTdat[:, 0], self.VPTdat[:, 1], markerfmt=' ', linefmt='k-',
                                                  use_line_collection=True)
        plt.setp(stemline, 'linewidth', 3.0)
        plt.setp(baseline, visible=False)
        ax1.set_ylim(0, 700)
        ax1.set_xlim(3000, 3750)
        ax1.set_ylabel('Intensity')
        ax1.set_xlabel('Energy ($\mathrm{cm}^{-1}$)')
        plt.tight_layout()
        plt.savefig(f"{self.SaveDir}/{self.filetag}OHStretchSticks.jpg")
        plt.close()

    def make_SB_sticks(self):
        plt.rcParams.update({'font.size': 20})
        fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=300)
        markerlline, stemlline, baselline = plt.stem(self.VPTdat[:, 0], self.VPTdat[:, 1], markerfmt=' ', linefmt='k-',
                                                     use_line_collection=True)
        plt.setp(stemlline, 'linewidth', 3.0)
        plt.setp(baselline, visible=False)
        ax2.set_ylim(0, 25)
        ax2.set_xlim(4500, 5400)
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Energy ($\mathrm{cm}^{-1}$)')
        plt.tight_layout()
        plt.savefig(f"{self.SaveDir}/{self.filetag}SBSticks.jpg")

class Plots:
    def __init__(self):
        self.rawData = np.loadtxt("SBdata_May27.csv", delimiter=",", skiprows=1, dtype=str)  # UPDATE IF DATA UPDATES
        self.DataHeaders = ["WaterNum", "Number of Acceptors", "Number of Donors", "OHO Angle (Degrees)",
                            "OO Distance ($\mathrm{\AA}$)", "Bend Frequency ($\mathrm{cm}^{-1}$)",
                            "Bend Intensity (km/mol)", "Stretch Frequency ($\mathrm{cm}^{-1}$)",
                            "Stretch Intensity (km/mol)", "Stretch-Bend Frequency ($\mathrm{cm}^{-1}$)",
                            "Stretch-Bend Intensity (km/mol)"]
        self.ColorDict = {"None": "gold",
                          "D": "purple",
                          "A": "darkgray",
                          "AD": "g",
                          "ADD": "darkorange",
                          "AA": "fuchsia",
                          "AAD": "crimson",
                          "AADD": "darkblue"}
        self.MarkerDict = {"Monomer": "o",
                           "Dimer": "^",
                           "Tet Cage": "P",
                           "Tet Three-One": "X",
                           "Pent Cage": "D",
                           "Pent Ring": "d",
                           "Hex Cage": "s"}
        self._FigDir = None
        self._DataDict = None

    @property
    def FigDir(self):
        if self._FigDir is None:
            docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self._FigDir = os.path.join(docs, "stretch_bend", "Figures")
        return self._FigDir

    @property
    def DataDict(self):
        if self._DataDict is None:
            self._DataDict = self.formatData()
        return self._DataDict

    @staticmethod
    def findHBType(OH_dat):
        if OH_dat[1] == 0 and OH_dat[2] == 0:
            HBtype = "None"
        elif OH_dat[1] == 0 and OH_dat[2] == 1:
            HBtype = "D"
        elif OH_dat[1] == 1 and OH_dat[2] == 0:
            HBtype = "A"
        elif OH_dat[1] == 1 and OH_dat[2] == 1:
            HBtype = "AD"
        elif OH_dat[1] == 1 and OH_dat[2] == 2:
            HBtype = "ADD"
        elif OH_dat[1] == 2 and OH_dat[2] == 0:
            HBtype = "AA"
        elif OH_dat[1] == 2 and OH_dat[2] == 1:
            HBtype = "AAD"
        elif OH_dat[1] == 2 and OH_dat[2] == 2:
            HBtype = "AADD"
        else:
            raise Exception(f"Can not determine H-bond environment for {OH_dat}")
        return HBtype

    def formatData(self):
        dataDict = dict()
        for struct in self.rawData:
            if struct[0] not in dataDict:
                dataDict[struct[0]] = []
            floatDat = [float(x) if x != "None" else -1.0 for x in struct[1:] if x != ""]
            dataDict[struct[0]].append(floatDat)
        data4real = dict()
        for key in dataDict:
            if len(key) <= 4:
                pass
            else:
                data4real[key] = np.array(dataDict[key])
        return data4real

    def plotOHOvsSBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=600)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                Htype = self.findHBType(OH)
                if "D" in Htype:
                    x = OH[3]
                    y = OH[-1]
                    plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[Htype], marker=self.MarkerDict[key],
                             markersize=10)
                    if key not in legend_markers:
                        legend_markers.append(key)
                    if Htype not in legend_colors:
                        legend_colors.append(Htype)
                else:
                    pass
        plt.xlim(120, 180)
        plt.ylim(0, 15)
        plt.xlabel(self.DataHeaders[3])
        plt.ylabel(self.DataHeaders[-1])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=self.MarkerDict[key], markerfacecolor='k', color='w',
                                         markersize=10, label=key))
        for Htype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[Htype], edgecolor=self.ColorDict[Htype], label=Htype))
        plt.legend(handles=legendElements, loc="upper left")
        plt.savefig(os.path.join(self.FigDir, "OHOvsSBI.png"), dpi=fig.dpi, bboxinches="tight")

    def plotOOvsSBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=600)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                Htype = self.findHBType(OH)
                if "D" in Htype:
                    x = OH[4]
                    y = OH[-1]
                    plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[Htype], marker=self.MarkerDict[key],
                             markersize=10)
                    if key not in legend_markers:
                        legend_markers.append(key)
                    if Htype not in legend_colors:
                        legend_colors.append(Htype)
                else:
                    pass
        plt.xlim(2.65, 3.1)
        plt.ylim(0, 15)
        plt.xlabel(self.DataHeaders[4])
        plt.ylabel(self.DataHeaders[-1])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=self.MarkerDict[key], markerfacecolor='k', color='w',
                                         markersize=10, label=key))
        for Htype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[Htype], edgecolor=self.ColorDict[Htype], label=Htype))
        plt.legend(handles=legendElements, loc="upper right")
        plt.savefig(os.path.join(self.FigDir, "OOvsSBI.png"), dpi=fig.dpi, bboxinches="tight")



