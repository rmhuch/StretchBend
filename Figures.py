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
            self._VPTdat = pull_VPTblock(self.ClusterObj.eqlog)[1]
        return self._VPTdat

    def make_OH_Sticks(self):
        plt.rcParams.update({'font.size': 20})
        fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
        markerline, stemline, baseline = plt.stem(self.VPTdat[:, 0], self.VPTdat[:, 1], markerfmt=' ', linefmt='b-',
                                                  use_line_collection=True)
        plt.setp(stemline, 'linewidth', 3.0)
        plt.setp(baseline, visible=False)
        # markerrline, stemmline, baseeline = plt.stem(self.VPTdat[13:, 0], self.VPTdat[13:, 1], markerfmt=' ', linefmt='k-',
        #                                          use_line_collection=True)
        # plt.setp(stemmline, 'linewidth', 3.0)
        # plt.setp(baseeline, visible=False)
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
        ax2.set_ylim(0, 30)
        ax2.set_xlim(4500, 5400)
        ax2.set_ylabel('Intensity')
        ax2.set_xlabel('Energy ($\mathrm{cm}^{-1}$)')
        plt.tight_layout()
        plt.savefig(f"{self.SaveDir}/{self.filetag}SBSticks.jpg")

class Plots:
    def __init__(self, DataSet):
        self.DataSet = DataSet  # string identifying data being plotted
        self.rawData = np.loadtxt("SBdata_Sept29.csv", delimiter=",", skiprows=1, dtype=str)  # UPDATE IF DATA UPDATES
        self.DataHeaders = ["WaterNum", "Number of Acceptors", "Number of Donors", "Number of Acceptors (HOH)",
                            "Number of Donors (HOH)", "OHO Angle (Degrees)", "OO Distance ($\mathrm{\AA}$)",
                            "Bend Frequency ($\mathrm{cm}^{-1}$)", "Bend Intensity (km/mol)",
                            "Stretch Frequency ($\mathrm{cm}^{-1}$)", "Stretch Intensity (km/mol)",
                            "Stretch-Bend Frequency ($\mathrm{cm}^{-1}$)", "Stretch-Bend Intensity (km/mol)",
                            "VPT2 Stretch-Bend Intensity (km/mol)", "Average Stretch-Bend Intensity (km/mol)",
                            r"$\mathrm{\frac{\partial ^2 \mu}{\partial r \partial \theta} / \frac{\partial \mu}{\partial r}}$"]
        self.ColorDict = {"None": "gold",
                          "D": "purple",
                          "A": "darkgray",
                          "AD": "g",
                          "ADD": "darkorange",
                          "AA": "fuchsia",
                          "AAD": "crimson",
                          "AADD": "darkblue"}
        # self.MarkerDict = {"Monomer": "o",  <-- original
        #                    "Dimer": "^",
        #                    "4-Cage": "P",
        #                    "4-Three-One": "X",
        #                    "5-Cage": "D",
        #                    "5-Ring": "d",
        #                    "6-Cage": "s"}
        self.MarkerDict = {"Monomer": "o",
                           "Dimer": "h",
                           "4-Cage": "P",
                           "4-Three-One": "P",
                           "5-Cage": "D",
                           "5-Ring": "D",
                           "6-Cage": "v",
                           "4+-Ring": "X",
                           "5+-Ring": "d",
                           "6+-T1": "^",
                           "4-Cage-TZ": "*"}
        self.LabelDict = {"o": "H$_{2}$O",  # used to call the labels for the legend
                          "h": "(H$_{2}$O)$_{2}$",
                          "P": "(H$_{2}$O)$_{4}$",
                          "D": "(H$_{2}$O)$_{5}$",
                          "v": "(H$_{2}$O)$_{6}$",
                          "X": "H$^{+}$(H$_{2}$O)$_{4}$",
                          "d": "H$^{+}$(H$_{2}$O)$_{5}$",
                          "^": "H$^{+}$(H$_{2}$O)$_{6}$",
                          "*": "(H$_{2}$O)$_{4}$ - TZ"}
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

    @staticmethod
    def findHOHType(OH_dat):
        if OH_dat[3] == 0 and OH_dat[4] == 0:
            HBtype = "None"
        elif OH_dat[3] == 0 and OH_dat[4] == 1:
            HBtype = "D"
        elif OH_dat[3] == 1 and OH_dat[4] == 0:
            HBtype = "A"
        elif OH_dat[3] == 1 and OH_dat[4] == 1:
            HBtype = "AD"
        elif OH_dat[3] == 1 and OH_dat[4] == 2:
            HBtype = "ADD"
        elif OH_dat[3] == 2 and OH_dat[4] == 0:
            HBtype = "AA"
        elif OH_dat[3] == 2 and OH_dat[4] == 1:
            HBtype = "AAD"
        elif OH_dat[3] == 2 and OH_dat[4] == 2:
            HBtype = "AADD"
        else:
            raise Exception(f"Can not determine HOH's H-bond environment for {OH_dat}")
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
        if self.DataSet == "All":  # include neutral and charged DZ clusters
            keys_to_delete = []
            for i in data4real.keys():
                if i.find("TZ") >= 0:
                    keys_to_delete.append(i)
                else:
                    pass
            [data4real.pop(key) for key in keys_to_delete]
        elif self.DataSet == "Neutral":  # if "+" IS in the key delete it (ONLY DZ clusters)
            keys_to_delete = []
            for i in data4real.keys():
                if i.find("+") >= 0 or i.find("TZ") >= 0:
                    keys_to_delete.append(i)
                else:
                    pass
            [data4real.pop(key) for key in keys_to_delete]
        elif self.DataSet == "Charged":
            keys_to_delete = []
            for i in data4real.keys():
                if i.find("+") <= 0:  # "if "+" isn't in the key delete it
                    keys_to_delete.append(i)
                else:
                    pass
            [data4real.pop(key) for key in keys_to_delete]
        elif self.DataSet == "DZvsTZ":
            keys_to_delete = []
            for i in data4real.keys():
                if i.find("4-Cage") < 0:  # if "4-Cage" isn't in the key delete it
                    keys_to_delete.append(i)
                else:
                    pass
            [data4real.pop(key) for key in keys_to_delete]
        else:
            raise Exception(f"Data set {self.DataSet} is not defined.")
        return data4real

    def plotOHOvsSBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                if "D" in HOHtype:
                    x = OH[5]
                    y = OH[12]
                    plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                             markersize=10)
                    if self.MarkerDict[key] not in legend_markers:
                        legend_markers.append(self.MarkerDict[key])
                    if HOHtype not in legend_colors:
                        legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(130, 180)
        plt.ylim(0, 20)
        plt.xlabel(self.DataHeaders[5])
        plt.ylabel(self.DataHeaders[12])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"OHOvsSBI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotOHOvsSI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                if "D" in HOHtype:
                    x = OH[5]
                    y = OH[10]
                    plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                             markersize=10)
                    if self.MarkerDict[key] not in legend_markers:
                        legend_markers.append(self.MarkerDict[key])
                    if HOHtype not in legend_colors:
                        legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(130, 180)
        plt.ylim(0, 1000)
        plt.xlabel(self.DataHeaders[5])
        plt.ylabel(self.DataHeaders[10])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"OHOvsSI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotOHOvsBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                if "D" in HOHtype:
                    x = OH[5]
                    y = OH[8]
                    plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                             markersize=10)
                    if self.MarkerDict[key] not in legend_markers:
                        legend_markers.append(self.MarkerDict[key])
                    if HOHtype not in legend_colors:
                        legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(130, 180)
        plt.ylim(0, 90)
        plt.xlabel(self.DataHeaders[5])
        plt.ylabel(self.DataHeaders[8])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"OHOvsBI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotOOvsSBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                if "D" in HOHtype:
                    x = OH[6]
                    y = OH[12]
                    plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                             markersize=10)
                    if self.MarkerDict[key] not in legend_markers:
                        legend_markers.append(self.MarkerDict[key])
                    if HOHtype not in legend_colors:
                        legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(2.65, 3.1)
        plt.ylim(0, 20)
        plt.xlabel(self.DataHeaders[6])
        plt.ylabel(self.DataHeaders[12])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"OOvsSBI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotOOvsSI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                if "D" in HOHtype:
                    x = OH[6]
                    y = OH[10]
                    plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                             markersize=10)
                    if self.MarkerDict[key] not in legend_markers:
                        legend_markers.append(self.MarkerDict[key])
                    if HOHtype not in legend_colors:
                        legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(2.65, 3.1)
        plt.ylim(0, 1000)
        plt.xlabel(self.DataHeaders[6])
        plt.ylabel(self.DataHeaders[10])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"OOvsSI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotOOvsBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                if "D" in HOHtype:
                    x = OH[6]
                    y = OH[8]
                    plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                             markersize=10)
                    if self.MarkerDict[key] not in legend_markers:
                        legend_markers.append(self.MarkerDict[key])
                    if HOHtype not in legend_colors:
                        legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(2.65, 3.1)
        plt.ylim(0, 90)
        plt.xlabel(self.DataHeaders[6])
        plt.ylabel(self.DataHeaders[8])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"OOvsBI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotSBfreqvsSBI(self, dataset="LM"):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                x = OH[11]
                if dataset == "LM":
                    y = OH[12]
                elif dataset == "VPT":
                    y = OH[13]
                else:
                    raise Exception(f"Can not determine dataset = {dataset}")
                plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                         markersize=10)
                if self.MarkerDict[key] not in legend_markers:
                    legend_markers.append(self.MarkerDict[key])
                if HOHtype not in legend_colors:
                    legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(4900, 5600)
        plt.ylim(0, 20)
        plt.xlabel(self.DataHeaders[11])
        plt.ylabel(self.DataHeaders[12])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"SBFreqvsSBI_{self.DataSet}_{dataset}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotSfreqvsSI(self):
        plt.rcParams.update({'font.size': 25})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                x = OH[9]
                y = OH[10]
                plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                         markersize=10)
                if self.MarkerDict[key] not in legend_markers:
                    legend_markers.append(self.MarkerDict[key])
                if HOHtype not in legend_colors:
                    legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(3200, 4000)
        plt.ylim(0, 1000)
        plt.xlabel(self.DataHeaders[9])
        plt.ylabel(self.DataHeaders[10])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))

        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"SFreqvsSI_{self.DataSet}_bigfont.png"), dpi=fig.dpi, bboxinches="tight")

    def plotBfreqvsBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=600)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                x = OH[7]
                y = OH[8]
                plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                         markersize=10)
                if self.MarkerDict[key] not in legend_markers:
                    legend_markers.append(self.MarkerDict[key])
                if HOHtype not in legend_colors:
                    legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(1620, 1720)
        plt.ylim(0, 90)
        plt.xlabel(self.DataHeaders[7])
        plt.ylabel(self.DataHeaders[8])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))

        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"BFreqvsBI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotVPTvsSBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                x = OH[13]
                y = OH[12]
                plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                         markersize=10)
                if self.MarkerDict[key] not in legend_markers:
                    legend_markers.append(self.MarkerDict[key])
                if HOHtype not in legend_colors:
                    legend_colors.append(HOHtype)
                else:
                    pass
        plt.plot(np.arange(0, 25), np.arange(0, 25), "-k", linewidth=1.0)
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.xlabel(self.DataHeaders[13])
        plt.ylabel(self.DataHeaders[12])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))

        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"VPT2vsSBI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotVPTvsSBI_AVG(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                x = OH[13]
                y = OH[14]
                plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                         markersize=10)
                if self.MarkerDict[key] not in legend_markers:
                    legend_markers.append(self.MarkerDict[key])
                if HOHtype not in legend_colors:
                    legend_colors.append(HOHtype)
                else:
                    pass
        plt.plot(np.arange(0, 25), np.arange(0, 25), "-k", linewidth=1.0)
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.xlabel(self.DataHeaders[13])
        plt.ylabel(self.DataHeaders[14])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"VPT2vsSBI_AVG_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotSIvsSBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                x = OH[10]
                y = OH[12]
                plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                         markersize=10)
                if self.MarkerDict[key] not in legend_markers:
                    legend_markers.append(self.MarkerDict[key])
                if HOHtype not in legend_colors:
                    legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(0, 1000)
        plt.ylim(0, 20)
        plt.xlabel(self.DataHeaders[10])
        plt.ylabel(self.DataHeaders[12])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))

        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"SIvsSBI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotDerivRatiovsSBI(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                x = OH[12]
                y = OH[15]
                plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                         markersize=10)
                if self.MarkerDict[key] not in legend_markers:
                    legend_markers.append(self.MarkerDict[key])
                if HOHtype not in legend_colors:
                    legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(0, 20)
        plt.ylim(0, 2.5)
        plt.xlabel(self.DataHeaders[12])
        plt.ylabel(self.DataHeaders[15])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"DerivRatiovsSBI_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")

    def plotDerivRatiovsVPT(self):
        plt.rcParams.update({'font.size': 20})
        legend_markers = []
        legend_colors = []
        fig = plt.figure(figsize=(12, 8), dpi=216)
        for key in self.DataDict:
            for OH in self.DataDict[key]:
                HOHtype = self.findHOHType(OH)
                x = OH[13]
                y = OH[15]
                plt.plot(x, y, color="k", markerfacecolor=self.ColorDict[HOHtype], marker=self.MarkerDict[key],
                         markersize=10)
                if self.MarkerDict[key] not in legend_markers:
                    legend_markers.append(self.MarkerDict[key])
                if HOHtype not in legend_colors:
                    legend_colors.append(HOHtype)
                else:
                    pass
        plt.xlim(0, 20)
        plt.ylim(0, 2.5)
        plt.xlabel(self.DataHeaders[13])
        plt.ylabel(self.DataHeaders[15])
        legendElements = []
        for key in legend_markers:
            legendElements.append(Line2D([0], [0], marker=key, markerfacecolor='k', color='w',
                                         markersize=10, label=self.LabelDict[key]))
        for HOHtype in legend_colors:
            legendElements.append(Patch(facecolor=self.ColorDict[HOHtype], edgecolor=self.ColorDict[HOHtype],
                                        label=HOHtype))
        plt.legend(handles=legendElements, bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.FigDir, f"DerivRatiovsVPT_{self.DataSet}.png"), dpi=fig.dpi, bboxinches="tight")