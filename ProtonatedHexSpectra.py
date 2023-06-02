import matplotlib.pyplot as plt
import numpy as np
import os
import glob

class PlotVPTSpect:
    def __init__(self, isdueterated=False, anharm=True, mixData=False, basis="tz", plot_sticks=True,
                 plot_convolutions=True, plot_exp=False, delta=5, region="full", isomer=None):
        self.isdueterated = isdueterated
        self.anharm = anharm  # DEFAULT IS VPT2 numbers, but pulls harmonics if False (and relabels figs)
        self.mixData = mixData  # Set anharm to false and mixData to true to plot VPT2 Freqs and Harm Intensities
        self.basis = basis
        self.plot_sticks = plot_sticks
        self.plot_convolutions = plot_convolutions
        self.plot_exp = plot_exp  # options: True, False, or "only"
        self.delta = delta
        self.region = region
        self.isomer = isomer
        self._colors = None
        self._MainDir = None
        self._logs = None
        self._DataSet = None
        self._expData = None

    @property
    def colors(self):
        if self._colors is None:
            self._colors = {"e2": "blue",
                            "t2": "green",
                            "z1": "red",
                            "t1": "lightseagreen",
                            "t4": "mediumspringgreen",
                            "p1": "darkorange",
                            "e1": "navy"}
        return self._colors

    @property
    def MainDir(self):
        if self._MainDir is None:
            docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self._MainDir = os.path.join(docs, "JacobHexamerVPT")
        return self._MainDir

    @property
    def logs(self):
        if self._logs is None:
            if self.isdueterated:
                self._logs = glob.glob(os.path.join(self.MainDir, self.basis, "d13o6p*.log"))
            else:
                self._logs = glob.glob(os.path.join(self.MainDir, self.basis, "h13o6p*.log"))
        return self._logs

    @property
    def DataSet(self):
        if self._DataSet is None:
            self._DataSet = self.get_DataSet()
        return self._DataSet

    @property
    def expData(self):
        if self._expData is None:
            if self.plot_exp == "only":
                self._expData = self.get_AllexpData()
            else:
                self._expData = self.get_expData()
        return self._expData

    def get_DataSet(self):
        from NMParser import pull_VPTblock
        data_dict = dict()  # this data dict holds data for ALL isomers
        for log in self.logs:
            harmDat, VPTdat = pull_VPTblock(log)
            fname = log.split("/")[-1]
            isomer = fname.split("_")[1]
            if self.anharm:
                data_dict[isomer] = VPTdat
            else:
                if self.mixData:
                    data_dict[isomer] = np.column_stack((VPTdat[:, 0], harmDat[:, 1]))
                else:
                    data_dict[isomer] = harmDat
        return data_dict

    def get_expData(self):
        """returns the x/y data for either D13O6 or H13O6 (dependent on `is_deuterated`) as an array to be plotted."""
        if self.isdueterated:
            dat_path = os.path.join(self.MainDir, "d13o6p_expSpect.dat")
            dat = np.loadtxt(dat_path, skiprows=1)
        else:
            dat1_path = os.path.join(self.MainDir, "spectrum_part1.dat")
            dat2_path = os.path.join(self.MainDir, "spectrum_part2.dat")
            dat1 = np.loadtxt(dat1_path)
            dat2 = np.loadtxt(dat2_path)
            dat = (dat1, dat2)
        return dat

    def get_AllexpData(self):
        """returns the x/y data for either D13O6 or H13O6 (dependent on `is_deuterated`) as an array to be plotted."""
        expDat = dict()
        allD_dat_path = os.path.join(self.MainDir, "d13o6p_expSpect.dat")
        expDat["allD"] = np.loadtxt(allD_dat_path, skiprows=1)
        dat1_path = os.path.join(self.MainDir, "spectrum_part1.dat")
        dat2_path = os.path.join(self.MainDir, "spectrum_part2.dat")
        expDat["allH1"] = np.loadtxt(dat1_path)
        expDat["allH2"] = np.loadtxt(dat2_path)
        return expDat

    def FindFigLabel(self):
        if self.isdueterated:  # do we have all H or all D?
            figlabel = "allD_hex_spect_"
        else:
            figlabel = "allH_hex_spect_"
        if len(self.isomer) == 1:
            figlabel += f"{self.isomer}"
        else:
            for iso in self.isomer:
                figlabel += f"{iso}"
        if self.anharm:  # are we plotting VPT results or harmonics
            figlabel += "_VPT"
        else:
            if self.mixData:
                figlabel += "_MixSpecialZ1"
            else:
                figlabel += "_Harm"
        if self.plot_convolutions and self.plot_sticks:  # what does the spectrum look like?
            figlabel += f"_stickConvolute_D{self.delta}"
        elif self.plot_sticks:
            figlabel += "_stick"
        elif self.plot_convolutions:
            figlabel += f"_Convolute_D{self.delta}"
        if self.plot_exp:
            figlabel += "_wExp"
        figlabel += f"_{self.region}_{self.basis}.png"
        return figlabel

    @staticmethod
    def plotSticks(ax, dat, color, lw):
        mkline, stline, baseline = ax.stem(dat[:, 0], dat[:, 1], linefmt=f"{color}",
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

    def plot_Spect(self):
        from matplotlib.patches import Patch
        plt.rcParams.update({'font.size': 14})
        if len(self.isomer) == 1:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3), dpi=216)
            plot_one = True
        else:
            fig, ax = plt.subplots(nrows=len(self.isomer), ncols=1, figsize=(6, 6),
                                   dpi=216, sharex="all", sharey="all")
            plot_one = False
        legendElements = []
        if plot_one:
            if self.plot_sticks:
                self.plotSticks(ax, self.DataSet[self.isomer[0]], self.colors[self.isomer[0]], 2)
            if self.plot_convolutions:
                self.plotGauss(ax, self.DataSet[self.isomer[0]], self.colors[self.isomer[0]], 2, delta=self.delta)
            if self.plot_exp:
                if self.isdueterated:
                    ax.plot(self.expData[:, 0], (self.expData[:, 1]*50)+100, "-k", linewidth=2.5, zorder=0)
                else:
                    for dat in self.expData:
                        ax.plot(dat[:, 0], dat[:, 1]+100, "-k", linewidth=2.5, zorder=0)
        else:
            for i, iso in enumerate(self.isomer):
                if self.mixData:
                    if iso == "z1":
                        if self.isdueterated:
                            self.DataSet[iso][9, 0] += 500
                        else:
                            self.DataSet[iso][9, 0] += 1100
                            self.DataSet[iso][11, 0] += 300
                if self.plot_sticks:
                    self.plotSticks(ax[i], self.DataSet[iso], self.colors[iso], 2)
                if self.plot_convolutions:
                    self.plotGauss(ax[i], self.DataSet[iso], self.colors[iso], 2, delta=self.delta)
                if self.plot_exp:
                    if self.isdueterated:
                        ax[i].plot(self.expData[:, 0], (self.expData[:, 1]*50)+100, "-k", linewidth=2.5, zorder=0)
                    else:
                        for dat in self.expData:
                            ax[i].plot(dat[:, 0], dat[:, 1] + 100, "-k", linewidth=2.5, zorder=0)
                ax[i].axes.get_yaxis().set_visible(False)
        plt.xlabel(r"Frequency ($\mathrm{cm}^{-1}$)")
        plt.ylabel("Intensity")
        # plt.legend(handles=legendElements, loc='upper right')
        if self.isdueterated:
            if self.region == "full":
                plt.ylim(0, 1500)
                plt.xlim(500, 3000)
            elif self.region == "Stretch":
                plt.ylim(0, 1500)
                plt.xlim(1642, 2941)
            elif self.region == "MJStretch":
                plt.ylim(0, 1500)
                plt.xlim(2200, 2800)
            elif self.region == "SB":
                plt.ylim(0, 15)
                plt.xlim(3000, 7000)
        else:
            if self.region == "full":
                plt.ylim(0, 1500)
                plt.xlim(800, 4000)
            elif self.region == "Stretch":
                plt.ylim(0, 1500)
                plt.xlim(2601, 3900)
            elif self.region == "SB":
                plt.ylim(0, 15)
                plt.xlim(4000, 8000)
        figlabel = self.FindFigLabel()
        figname = os.path.join(self.MainDir, "Spectra", figlabel)
        plt.tight_layout()
        plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
        plt.close()

    def plot_ExpSpect(self):
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), dpi=216)
        # plot all H on top axis
        ax[0].plot(self.expData["allH1"][:, 0], self.expData["allH1"][:, 1] + 100, "-k", linewidth=2.5, zorder=0)
        ax[0].plot(self.expData["allH2"][:, 0], self.expData["allH2"][:, 1] + 100, "-k", linewidth=2.5, zorder=0)
        ax[0].text(2650, 1161, r"$\mathrm{H^+(H_2O)_6}$", ha="left", va="top")
        ax[0].axes.set_xlim(2601, 3900)
        ax[0].axes.get_yaxis().set_visible(False)
        # plot all D on bottom axis
        ax[1].plot(self.expData["allD"][:, 0], self.expData["allD"][:, 1], "-k", linewidth=2.5, zorder=0)
        ax[1].text(1691, 22, r"$\mathrm{D^+(D_2O)_6}$", ha="left", va="top")
        ax[1].axes.set_xlim(1642, 2941)
        ax[1].axes.get_yaxis().set_visible(False)
        plt.xlabel(r"Frequency ($\mathrm{cm}^{-1}$)")
        figlabel = "ExpOnly_Stretch_allD_allH"
        figname = os.path.join(self.MainDir, "Spectra", figlabel)
        plt.tight_layout()
        plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
        plt.close()
