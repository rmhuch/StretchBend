import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

class SpectraPlot:
    def __init__(self, cluster_size, isomer, coupling, transition="SB", plot_sticks=True, plot_convolutions=True,
                 delta=5):
        self.cluster_size = cluster_size
        self.isomer = isomer  # MUST SET AS `None` TO RUN W2!
        self.coupling = coupling
        self.transition = transition

        self.plot_sticks = plot_sticks
        self.plot_convolutions = plot_convolutions
        self.delta = delta
        self._colors = None
        self._linetypes = None
        self._ClusterDir = None
        self._DataSet = None

    @property
    def colors(self):
        if self._colors is None:
            self._colors = {"lm": "c",  # local mode, only diagonal of fg matrix
                            "OH": "m",  # all OH stretches couple
                            "bnd": "y",  # all HOH bends couple
                            "HOH": "r",  # intramolecular modes of one H2O couple (S/S/B of each water)
                            "intra": "g",  # all high frequency modes couple with each other
                            "nm": "b"}  # normal mode, entire fg matrix
        return self._colors

    @property
    def linetypes(self):
        if self._linetypes is None:
            self._linetypes = {"AD": "-",  # only dimer
                               "ring": "-", # w4, w6
                               "cage": "--",  # w4, w6
                               "3_1": "-.",  # only w4
                               "prism": "-.",  # only w6
                               "book": ":"}  # only w6
        return self._linetypes

    @property
    def ClusterDir(self):
        if self._ClusterDir is None:
            docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            MainDir = os.path.join(docs, "stretch_bend", "AnneLMcode")
            self._ClusterDir = os.path.join(MainDir, f"w{self.cluster_size}")
        return self._ClusterDir

    @property
    def DataSet(self):
        """Data is stored in pickle (pkl) file format as a dictionary. There is a pkl file for each isomer containing
        the fundamental and SB for each of the 6 methods. If class has multiple isomers, data set becomes nested dict"""
        if self._DataSet is None:
            if len(self.isomer) > 1:  # plotting multiple isomers
                DataSet = dict()
                for iso in self.isomer:
                    f_path = os.path.join(self.ClusterDir, iso, "SpectraDict.pkl")
                    if os.path.exists(f_path):
                        with open(f_path, "rb") as fp:
                            DataSet[iso] = pickle.load(fp)
                    else:
                        DataSet[iso] = self.pullData(iso)
            else:  # plotting one isomer
                if self.isomer[0] == "AD":  # pulls dimer data
                    f_path = os.path.join(self.ClusterDir, "SpectraDict.pkl")
                else:
                    f_path = os.path.join(self.ClusterDir, self.isomer[0], "SpectraDict.pkl")
                if os.path.exists(f_path):
                    with open(f_path, "rb") as fp:
                        DataSet = pickle.load(fp)
                else:
                    DataSet = self.pullData(self.isomer[0])
            self._DataSet = DataSet
        return self._DataSet

    def pullData(self, iso=None):
        """this will pull and store the frequency/intensity data from every calculation type for one isomer
         and saves it to a pickle file"""
        data_dict = dict()
        for calc_type in ["lm", "nm", "intra", "bnd", "HOH", "OH"]:
            # pull all the data for one type of calculation
            if self.cluster_size == 2:
                all_freqs = np.loadtxt(os.path.join(self.ClusterDir, f"{calc_type}_freq.dat"))
                SB_intensities = np.loadtxt(os.path.join(self.ClusterDir, f"{calc_type}_SB.dat"))
                fund_intensities = np.loadtxt(os.path.join(self.ClusterDir, f"{calc_type}_lin_dip.dat"))
                f_path = os.path.join(self.ClusterDir, "SpectraDict.pkl")  # where the data will save
            else:
                all_freqs = np.loadtxt(os.path.join(self.ClusterDir, iso, f"{calc_type}_freq.dat"))
                SB_intensities = np.loadtxt(os.path.join(self.ClusterDir, iso, f"{calc_type}_SB.dat"))
                fund_intensities = np.loadtxt(os.path.join(self.ClusterDir, iso, f"{calc_type}_lin_dip.dat"))
                f_path = os.path.join(self.ClusterDir, iso, "SpectraDict.pkl")  # where the data will save
            # format STRETCH - pull all stretch modes starting after bends remember: # bends = # H2O
            Fmodes = all_freqs[self.cluster_size:, 0]
            Ffreqs = np.zeros(len(Fmodes))  # create arrays to dump freq/intensity in
            Fints = np.zeros(len(Fmodes))
            for i, mode in enumerate(Fmodes):
                Ffreqs[i] = all_freqs[np.argwhere(all_freqs[:, 0] == mode), 1]
                Fints[i] = fund_intensities[np.argwhere(fund_intensities[:, 0] == mode), -1]
            format_fund_dat = np.column_stack((Ffreqs, Fints))
            # format STRETCH BEND
            SBmodes = SB_intensities[:, :2]
            # for the intensities pulled, grab the frequency (add two contributing modes together)
            transitionF = np.zeros(len(SBmodes))
            for i, mode in enumerate(SBmodes):
                freq1 = all_freqs[np.argwhere(all_freqs[:, 0] == mode[0]), 1]
                freq2 = all_freqs[np.argwhere(all_freqs[:, 0] == mode[1]), 1]
                transitionF[i] = float(freq1) + float(freq2)
            format_SB_dat = np.column_stack((transitionF, SB_intensities[:, -1]))
            data_dict[calc_type] = np.row_stack((format_fund_dat, format_SB_dat))
        with open(f_path, "wb") as fp:
            pickle.dump(data_dict, fp)
        print(f"Data saved to {f_path}")
        return data_dict

    @staticmethod
    def plotSticks(ax, dat, color, linetype, lw):
        mkline, stline, baseline = ax.stem(dat[:, 0], dat[:, 1], linefmt=f"{linetype}{color}",
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
        figlabel = f"w{self.cluster_size}_{self.transition}_"
        for iso in self.isomer:
            figlabel += f"{iso}"
        figlabel += "_"
        for coup in self.coupling:
            figlabel += f"{coup}"
        figlabel += "_"
        if self.plot_sticks and self.plot_convolutions:
            figlabel += f"stickConvolute_D{self.delta}.png"
        elif self.plot_sticks:
            figlabel += f"stick.png"
        elif self.plot_convolutions:
            figlabel += f"convolute_D{self.delta}.png"
        return figlabel

    def plot_Spect(self):
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        plt.rcParams.update({'font.size': 14})
        legendElements = []
        # determine number of plots called for
        if len(self.isomer) == 1 and len(self.coupling) == 1:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), dpi=216)
            plot_one = True
        elif len(self.isomer) == 1 and len(self.coupling) > 1:  # plots one isomer for multiple coupling types
            fig, ax = plt.subplots(nrows=len(self.coupling), ncols=1, figsize=(8, 8), dpi=216,
                                   sharex="all", sharey="all")
            plot_one = False
        elif len(self.isomer) > 1 and len(self.coupling) == 1:  # plots multiple isomers for one coupling type
            fig, ax = plt.subplots(nrows=len(self.isomer), ncols=1, figsize=(8, 8), dpi=216,
                                   sharex="all", sharey="all")
            plot_one = False
        elif len(self.isomer) > 1 and len(self.coupling) > 1:
            # plot multiple isomers for each coupling type (all isomers on each axis, each axis a different coupling)
            fig, ax = plt.subplots(nrows=len(self.coupling), ncols=1, figsize=(8, 8), dpi=216,
                                   sharex="all", sharey="all")
            plot_one = False
        else:
            raise Exception(f"unknown subplot arrangement for {self.isomer} and {self.coupling}")
        if plot_one:  # makes one plot: one isomer & one coupling method
            isomer = self.isomer[0]
            coupling = self.coupling[0]
            plt.suptitle(f"W{self.cluster_size} {isomer} {coupling}")
            if self.plot_sticks:
                self.plotSticks(ax, self.DataSet[coupling], "k", self.linetypes[isomer], 2)
            if self.plot_convolutions:
                self.plotGauss(ax, self.DataSet[coupling], self.colors[coupling], 2, delta=self.delta)
        else:
            if len(self.isomer) > 1:  # self.DataSet will be nested dict...
                if len(self.coupling) > 1:  # plot multiple isomers (on one ax) for each coupling method (subplot)
                    for i, coup in enumerate(self.coupling):
                        allData = []
                        for j, iso in enumerate(self.isomer):
                            if self.plot_sticks:
                                self.plotSticks(ax[i], self.DataSet[iso][coup], "k", self.linetypes[iso], 2)
                                if i == 0:
                                    legendElements.append(Line2D([0], [0], color="k", linewidth=2,
                                                             linestyle=self.linetypes[iso], label=iso))
                            if self.plot_convolutions:
                                allData.append(self.DataSet[iso][coup])
                        self.plotGauss(ax[i], np.concatenate(allData), self.colors[coup], 1.5, delta=self.delta)
                        legendElements.append(Patch(facecolor=self.colors[coup], label=coup))
                elif len(self.coupling) == 1:  # multiple isomers (subplot) for one coupling method
                    coupling = self.coupling[0]
                    plt.suptitle(f"W{self.cluster_size} {coupling}")
                    for j, iso in enumerate(self.isomer):
                        if self.plot_sticks:
                            self.plotSticks(ax[j], self.DataSet[iso][coupling], "k", self.linetypes[iso], 2)
                            legendElements.append(Line2D([0], [0], color="k", linewidth=2,
                                                         linestyle=self.linetypes[iso], label=iso))
                        if self.plot_convolutions:
                            self.plotGauss(ax[j],self.DataSet[iso][coupling], self.colors[coupling], 1.5,
                                           delta=self.delta)
            else:  # len(self.isomer == 1) & len(self.coupling > 1)
                isomer = self.isomer[0]
                for i, coup in enumerate(self.coupling):  # plot multiple coupling methods (subplot) for one isomer
                    plt.suptitle(f"W{self.cluster_size} {isomer}")
                    if self.plot_sticks:
                        self.plotSticks(ax[i], self.DataSet[coup], "k", self.linetypes[isomer], 2)
                    if self.plot_convolutions:
                        self.plotGauss(ax[i], self.DataSet[coup], self.colors[coup], 1.5, delta=self.delta)
                        legendElements.append(Patch(facecolor=self.colors[coup], label=coup))
        if len(legendElements) > 0:  # add legend if needed
            plt.figlegend(handles=legendElements, loc='center right')
        if self.transition == "SB":  # set axis limits
            if self.cluster_size == 6:
                plt.ylim(0, 25)
                plt.xlim(4600, 5650)
            elif self.cluster_size == 4:
                plt.ylim(0, 30)
                plt.xlim(4950, 5650)
            elif self.cluster_size == 2:
                plt.ylim(0, 15)
                plt.xlim(4950, 5650)
        if self.transition == "Fundamental":
            if self.cluster_size == 6:
                plt.ylim(0, 600)
                plt.xlim(3200, 4000)
            elif self.cluster_size == 4:
                plt.ylim(0, 600)
                plt.xlim(3200, 4000)
            elif self.cluster_size == 2:
                plt.ylim(0, 300)
                plt.xlim(3200, 4000)
        plt.subplots_adjust(left=0.125, right=0.825, top=0.9, hspace=0.25, wspace=0.25)  # adjust plots for axis labels
        fig.text(0.5, 0.04, r"Frequency ($\mathrm{cm}^{-1}$)", ha='center')  # add axis labels
        fig.text(0.04, 0.5, "Intensity (km/mol)", va='center', rotation='vertical')
        figlabel = self.defineFigLabel()
        figname = os.path.join(self.ClusterDir, "Spectra", figlabel)  # create path to cave fig
        plt.savefig(figname, dpi=fig.dpi, bboxinches="tight")
        plt.close()
