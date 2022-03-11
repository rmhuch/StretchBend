import numpy as np
from BendAxes import calcAxes
from NMParser import format_disps
import os
import csv

def calcWaterContribs(logfile, water_pos):
    """
    Calculates the NM projection for each H atom onto the axes (OH bond, perp. to HOH, in-plane HOH)
    :param logfile: path to Gaussian file (Opt/VPT2 of given isotopologue)
    :type logfile:
    :param water_pos: positions of the atoms that make the water molecule
    :type water_pos: list
    :return: in two arrays: H1 & H2 projections for each normal mode
    :rtype:tuple of np arrays
    """
    NMdisps = format_disps(logfile)
    axH1, axH2 = calcAxes(logfile, water_pos)
    H1_disps = NMdisps[:, water_pos[1], :]
    H1_proj = np.dot(H1_disps, axH1)
    H2_disps = NMdisps[:, water_pos[2], :]
    H2_proj = np.dot(H2_disps, axH2)
    return H1_proj, H2_proj

def writeResults(resfile, logfile, water_pos):
    H1, H2 = calcWaterContribs(logfile, water_pos)
    with open(resfile, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(np.column_stack((H1, H2)))


if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "tetramer_16", "three_one")
    f1 = os.path.join(MoleculeDir, "w4t_Hw1.log")
    f2 = os.path.join(MoleculeDir, "w4t_Hw2.log")
    f3 = os.path.join(MoleculeDir, "w4t_Hw3.log")
    f4 = os.path.join(MoleculeDir, "w4t_Hw4.log")
    w1 = [0, 3, 4]
    w2 = [1, 5, 6]
    w3 = [2, 7, 8]
    w4 = [9, 10, 11]
    writeResults(os.path.join(MoleculeDir, "ProjectionDisps_w1_H.csv"), f1, w1)
    writeResults(os.path.join(MoleculeDir, "ProjectionDisps_w2_H.csv"), f2, w2)
    writeResults(os.path.join(MoleculeDir, "ProjectionDisps_w3_H.csv"), f3, w3)
    writeResults(os.path.join(MoleculeDir, "ProjectionDisps_w4_H.csv"), f4, w4)
