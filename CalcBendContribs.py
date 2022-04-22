import numpy as np
from BendAxes import calcAxes
from NMParser import format_disps
from NormalModes import calc_disps
import os
import csv

def calcWaterContribs(water_pos, logfile=None, fchkfile=None):
    """
    Calculates the NM projection for each H atom onto the axes (OH bond, perp. to HOH, in-plane HOH)
    :param water_pos: positions of the atoms that make the water molecule
    :type water_pos: list
    :param logfile: path to Gaussian log file (Opt/VPT2 of given isotopologue)
    :type logfile:
    :param fchkfile: path to Gaussian fchk file (if calcing NM disps by hand)
    :type fchkfile:
    :return: in two arrays: H1 & H2 projections for each normal mode
    :rtype:tuple of np arrays
    """
    if logfile is not None:
        NMdisps = format_disps(logfile)
    elif fchkfile is not None:
        NMdisps = calc_disps(fchkfile, water_pos)
    else:
        raise Exception(f"Can not calculate with logfile = {logfile} and fchkfile = {fchkfile}")
    axH1, axH2 = calcAxes(logfile, water_pos)
    H1_disps = NMdisps[:, water_pos[1], :]
    H1_proj = np.dot(H1_disps, axH1)
    H2_disps = NMdisps[:, water_pos[2], :]
    H2_proj = np.dot(H2_disps, axH2)
    return H1_proj, H2_proj

def calcBendScaling(logfile, water_pos):
    H1_proj, H2_proj = calcWaterContribs(logfile, water_pos)
    H1val = abs(H1_proj[21, 2])  # WATCH THIS!! this index is SYSTEM dependent - 21 for tet, 7 for di, 2 for monomer
    # print(H1val)
    H2val = abs(H2_proj[21, 2])
    # print(H2val)
    norm_proj = H1val + H2val
    h1scale = H1val / norm_proj
    h2scale = H2val / norm_proj
    return h1scale, h2scale

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
