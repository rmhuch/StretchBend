import numpy as np
import os
import matplotlib.pyplot as plt
from NMParser import pull_VPTblock

def make_OH_Sticks(SaveDir, filetag, VPTdat):
    plt.rcParams.update({'font.size': 20})
    fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
    markerline, stemline, baseline = plt.stem(VPTdat[:, 0], VPTdat[:, 1], markerfmt=' ', linefmt='k-',
                                              use_line_collection=True)
    plt.setp(stemline, 'linewidth', 3.0)
    plt.setp(baseline, visible=False)
    ax1.set_ylim(0, 700)
    ax1.set_xlim(3000, 3750)
    ax1.set_ylabel('Intensity')
    ax1.set_xlabel('Energy ($\mathrm{cm}^{-1}$)')
    plt.tight_layout()
    plt.savefig(f"{SaveDir}/{filetag}OHStretchSticks.jpg")
    plt.close()

def make_SB_sticks(SaveDir, filetag, VPTdat):
    plt.rcParams.update({'font.size': 20})
    fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=300)
    markerlline, stemlline, baselline = plt.stem(VPTdat[:, 0], VPTdat[:, 1], markerfmt=' ', linefmt='k-',
                                                 use_line_collection=True)
    plt.setp(stemlline, 'linewidth', 3.0)
    plt.setp(baselline, visible=False)
    ax2.set_ylim(0, 25)
    ax2.set_xlim(4500, 5400)
    ax2.set_ylabel('Intensity')
    ax2.set_xlabel('Energy ($\mathrm{cm}^{-1}$)')
    plt.tight_layout()
    plt.savefig(f"{SaveDir}/{filetag}SBSticks.jpg")


if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "hexamer_dz", "cage")
    f1 = os.path.join(MoleculeDir, "w6c_Hw1.log")
    f2 = os.path.join(MoleculeDir, "w6c_Hw2.log")
    f3 = os.path.join(MoleculeDir, "w6c_Hw3.log")
    f4 = os.path.join(MoleculeDir, "w6c_Hw4.log")
    f5 = os.path.join(MoleculeDir, "w6c_Hw5.log")
    f6 = os.path.join(MoleculeDir, "w6c_Hw6.log")
    fALL = os.path.join(MoleculeDir, "w6c_allH.log")
    a, VPTdat1 = pull_VPTblock(f1)
    a2, VPTdat2 = pull_VPTblock(f2)
    a3, VPTdat3 = pull_VPTblock(f3)
    a4, VPTdat4 = pull_VPTblock(f4)
    a5, VPTdat5 = pull_VPTblock(f5)
    a6, VPTdat6 = pull_VPTblock(f6)
    aa, VPTallH = pull_VPTblock(fALL)
    allVPTdat = np.concatenate((VPTdat1, VPTdat2, VPTdat3, VPTdat4, VPTdat5, VPTdat6))
    make_OH_Sticks(MoleculeDir, "w6Water5", VPTdat5)
    make_SB_sticks(MoleculeDir, "w6Water5", VPTdat5)

