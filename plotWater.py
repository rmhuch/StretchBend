import numpy as np
import os
import matplotlib.pyplot as plt

"""This script loads in a csv file and plots the water spectrum as is used in the Introduction of SB2.
This data has been compiled from G.M. Hale and M.R. Querry, 
"Optical constants of water in the 200 nm to 200 micron wavelength region", Appl. Opt., 12, 555-563, (1973)."""
def load_data():
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filepath = os.path.join(docs, "stretch_bend", "WaterSpectrumXY.csv")
    alldata = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return alldata

def make_plot():
    dat = load_data()
    fig = plt.figure(figsize=(8, 8), dpi=216)
    # plot full spectrum
    ax1 = plt.subplot(212)
    ax1.plot(dat[:, 0], dat[:, 1], "-k")
    ax1.set_xlim(0, 6000)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_yaxis().set_visible(False)

    # plot stretch bend blow up
    sb_ind = np.argwhere(np.logical_and(dat[:, 0]>=4500, dat[:, 0]<=6000))
    ax3 = plt.subplot(222, sharey=ax1)
    ax3.plot(dat[sb_ind, 0], dat[sb_ind, 1]*20, "-g")
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.get_yaxis().set_visible(False)

    # plot association blow up
    assoc_ind = np.argwhere(np.logical_and(dat[:, 0]>=1850, dat[:, 0]<=2750))
    ax2 = plt.subplot(221, sharey=ax3)
    ax2.plot(dat[assoc_ind, 0], dat[assoc_ind, 1]*20, "-b")
    ax2.set_xlim(1500, 3000)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.get_yaxis().set_visible(False)

    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    figpath = os.path.join(docs, "stretch_bend", "WaterSpectrum.png")
    plt.savefig(figpath, dpi=fig.dpi, bboxinches="tight")

if __name__ == '__main__':
    make_plot()