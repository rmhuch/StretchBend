import os
import numpy as np
import matplotlib.pyplot as plt

class SpectaPlot:
    def __init__(self, cluster_size, isomer, transition="SB", method=None, plot_sticks=True, plot_convolutions=True,
                 delta=5):
        self.cluster_size = cluster_size
        self.isomer = isomer
        self.transition = transition
        if isinstance(self.isomer, str):
            self.DataFlag = "allmethods"
            self.method = None
        elif isinstance(self.isomer, list):
            self.DataFlag = "stackonemethod"
            self.method = method  # this is the one calculation type that will be pulled and plotted.
        else:
            raise Exception("Can not determine type of data to use for plots")
        self.plot_sticks = plot_sticks
        self.plot_convolutions = plot_convolutions
        self.delta = delta
        self._colors = None
        self._ClusterDir = None
        self._DataSet = None

    @property
    def colors(self):
        if self._colors is None:
            self._colors = ["c", "m", "r", "g", "b"]  # ["C0", "C3", "C4", "C5"]
            # if self.DataFlag == "allmethods":
            #     self._colors = ["C0", "C3", "C4", "C5"]
            # elif self.DataFlag == "stackonemethod":
            #     if self.method == "lm":
            #         self._colors = ["teal", "mediumspringgreen", "cornflowerblue", "darkcyan"]
            #     elif self.method == "HOH":
            #         self._colors = ["crimson", "orangered", "palevioletred", "brick"]
            #     elif self.method == "intra":
            #         self._colors = ["rebeccapurple", "darkviolet", "orchid", "violet"]
            #     elif self.method == "nm":
            #         self._colors = ["saddlebrown", "orange", "peru", "darkgoldenrod"]
            #     else:
            #         raise Exception(f"colors for cluster size {self.cluster_size} not defined")
            # else:
            #     raise Exception(f"colors for data flag {self.DataFlag} not defined")
        return self._colors

    @property
    def ClusterDir(self):
        if self._ClusterDir is None:
            docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            MainDir = os.path.join(docs, "stretch_bend", "AnneLMcode")
            self._ClusterDir = os.path.join(MainDir, f"w{self.cluster_size}")
        return self._ClusterDir

    @property
    def DataSet(self):
        if self._DataSet is None:
            if self.DataFlag == "allmethods":
                self._DataSet = self.pullData1()
            elif self.DataFlag == "stackonemethod":
                if self.transition == "SB":
                    self._DataSet = self.pullData2()
                elif self.transition == "Fundamental":
                    self._DataSet = self.pullDataF()
                else:
                    raise Exception(f"Can not interpret {self.transition} as transition type")
        return self._DataSet

    def pullData1(self):
        """this will pull and store the frequency/intensity data from every calculation type for the SB region
        into a dict (saved as self.DataSet)"""
        data_dict = dict()
        for type in ["lm", "nm", "intra", "HOH", "OH"]:
            # pull all the data for one type of calculation
            if self.cluster_size == 2:
                all_freqs = np.loadtxt(os.path.join(self.ClusterDir, f"{type}_freq.dat"))
                all_intensities = np.loadtxt(os.path.join(self.ClusterDir, f"{type}_SB.dat"))
            else:
                all_freqs = np.loadtxt(os.path.join(self.ClusterDir, self.isomer, f"{type}_freq.dat"))
                all_intensities = np.loadtxt(os.path.join(self.ClusterDir, self.isomer, f"{type}_SB.dat"))
            modes = all_intensities[:, :2]
            intensities = all_intensities[:, -1]
            # for the intensities pulled, grab the frequency (add two contributing modes together)
            transitionF = np.zeros(len(modes))
            for i, transition in enumerate(modes):
                freq1 = all_freqs[np.argwhere(all_freqs[:, 0] == transition[0]), 1]
                freq2 = all_freqs[np.argwhere(all_freqs[:, 0] == transition[1]), 1]
                transitionF[i] = float(freq1) + float(freq2)
            format_dat = np.column_stack((transitionF, intensities))
            data_dict[type] = format_dat
        return data_dict

    def pullData2(self):
        """this will pull and store the frequency/intensity data from one calculation type for every isomer for
        the SB region into a dict (saved as self.DataSet)"""
        data_dict = dict()
        for iso in self.isomer:  # loop through the isomers given
            # pull all the data for one type of calculation
            all_freqs = np.loadtxt(os.path.join(self.ClusterDir, iso, f"{self.method}_freq.dat"))
            all_intensities = np.loadtxt(os.path.join(self.ClusterDir, iso, f"{self.method}_SB.dat"))
            modes = all_intensities[:, :2]
            intensities = all_intensities[:, -1]
            # for the intensities pulled, grab the frequency (add two contributing modes together)
            transitionF = np.zeros(len(modes))
            for i, transition in enumerate(modes):
                freq1 = all_freqs[np.argwhere(all_freqs[:, 0] == transition[0]), 1]
                freq2 = all_freqs[np.argwhere(all_freqs[:, 0] == transition[1]), 1]
                transitionF[i] = float(freq1) + float(freq2)
            format_dat = np.column_stack((transitionF, intensities))
            data_dict[iso] = format_dat
        return data_dict

    def pullDataF(self):
        """this will pull and store the frequency/intensity data from one calculation type for every isomer for
        the STRETCH region into a dict (saved as self.DataSet)"""
        data_dict = dict()
        for iso in self.isomer:  # loop through the isomers given
            # pull all the data for one type of calculation
            all_freqs = np.loadtxt(os.path.join(self.ClusterDir, iso, f"{self.method}_freq.dat"))
            all_intensities = np.loadtxt(os.path.join(self.ClusterDir, iso, f"{self.method}_lin_dip.dat"))
            modes = all_intensities[:, :1].flatten()
            # for the intensities pulled, grab the frequency (add two contributing modes together)
            start_idx = np.argwhere(modes == all_freqs[4, 0])  # will only pull stretches
            format_dat = np.column_stack((all_freqs[4:, 1], all_intensities[int(start_idx):, -1]))
            data_dict[iso] = format_dat
        return data_dict

    @staticmethod
    def plotSticks(ax, dat, color, lw):
        mkline, stline, baseline = ax.stem(dat[:, 0], dat[:, 1], linefmt=f"-{color}",
                                            markerfmt=' ', use_line_collection=True)
        plt.setp(stline, "linewidth", lw)
        plt.setp(baseline, visible=False)

    @staticmethod
    def plotGauss(ax, dat, color, lw, delta):
        x = np.arange(min(dat[:, 0])-100, max(dat[:, 0])+100, 1)
        y = np.zeros(len(x))
        for i in np.arange(len(dat[:, 1])):
            for j, val in enumerate(x):
                y[j] += dat[i, 1]*np.exp(-(val-dat[i, 0])**2/delta**2)
        ax.plot(x, y, "-", color=color, linewidth=lw)

    def defineFigLabel(self):
        figlabel = "w"
        if self.cluster_size == 2:
            figlabel += f"{self.cluster_size}_{self.transition}_"
        elif isinstance(self.isomer, list):
            figlabel += f"{self.cluster_size}_{self.transition}_allISO_"
        elif isinstance(self.isomer, str):
            figlabel = f"w{self.cluster_size}_{self.transition}_{self.isomer}_"
        else:
            raise Exception("Can not name file")
        if self.plot_sticks and self.plot_convolutions:
            figlabel += f"stickConvolute_D{self.delta}.png"
        elif self.plot_sticks:
            figlabel += f"stick.png"
        elif self.plot_convolutions:
            figlabel += f"convolute_D{self.delta}.png"
        return figlabel

    def makeAllMethodsPlot(self):
        plt.rcParams.update({'font.size': 18})
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 10), dpi=216, sharex="all", sharey="all")
        plotTypes = ["lm", "OH", "HOH", "intra", "nm"]
        for i, ax in enumerate(axes):
            if self.plot_sticks:
                self.plotSticks(ax, self.DataSet[plotTypes[i]], self.colors[i], lw=4)
            if self.plot_convolutions:
                self.plotGauss(ax, self.DataSet[plotTypes[i]], "k", lw=2.5, delta=self.delta)
            ax.set_ylim(0, 10)
        plt.suptitle(f"W{self.cluster_size} {self.isomer}")
        plt.subplots_adjust(left=0.15, top=0.9, bottom=0.15, hspace=0.25, wspace=0.25)
        fig.text(0.5, 0.04, r"Frequency ($\mathrm{cm}^{-1}$)", ha='center')
        fig.text(0.04, 0.5, "Intensity", va='center', rotation='vertical')
        figlabel = self.defineFigLabel()
        figname = os.path.join(self.ClusterDir, "Spectra", figlabel)
        plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
        plt.close()

    def makeStackPlot(self):
        from matplotlib.patches import Patch
        # plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=216)
        legendElements = []
        allData = []
        for i, iso in enumerate(self.isomer):
            if self.plot_sticks:
                self.plotSticks(ax, self.DataSet[iso], "k", 2)
            if self.plot_convolutions:
                allData.append(self.DataSet[iso])
            legendElements.append(Patch(facecolor=self.colors[i], label=iso))
        if self.method == "lm":
            self.plotGauss(ax, np.concatenate(allData), self.colors[0], 2, delta=self.delta)
        elif self.method == "HOH":
            self.plotGauss(ax, np.concatenate(allData), self.colors[1], 2, delta=self.delta)
        elif self.method == "intra":
            self.plotGauss(ax, np.concatenate(allData), self.colors[2], 2, delta=self.delta)
        elif self.method == "nm":
            self.plotGauss(ax, np.concatenate(allData), self.colors[3], 2, delta=self.delta)
        else:
            raise Exception(f"can not determine color for method {self.method}")
        plt.xlabel(r"Frequency ($\mathrm{cm}^{-1}$)")
        plt.ylabel("Intensity")
        if self.transition == "SB":
            if self.cluster_size == 6:
                plt.ylim(0, 25)
            elif self.cluster_size == 4:
                plt.ylim(0, 30)
            elif self.cluster_size == 2:
                plt.ylim(0, 15)
        if self.transition == "Fundamental":
            if self.cluster_size == 4:
                plt.ylim(0, 3500)
                plt.xlim(3300, 4000)
            elif self.cluster_size == 6:
                plt.ylim(0, 6000)
                plt.xlim(3200, 4000)
        # plt.legend(handles=legendElements, loc='upper right')
        plt.tight_layout()
        figlabel = self.defineFigLabel()
        figname = os.path.join(self.ClusterDir, "Spectra", figlabel)
        plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
        plt.close()

    def makePlot(self):
        if self.DataFlag == "allmethods":
            self.makeAllMethodsPlot()
        elif self.DataFlag == "stackonemethod":
            self.makeStackPlot()
