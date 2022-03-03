from McUtils.GaussianInterface import GaussianLogReader
import numpy as np
import os
import csv

def pullStandardCoordinates(logfile):
    """
    This function reads a log file and pulls out the "Standard Coordinates" (optimized Cartesian Coordinates)
    :return: array of x y z coordinates for each atom
    :rtype: array
    """
    with GaussianLogReader(logfile) as reader:
        parse = reader.parse("StandardCartesianCoordinates")
    coordies = parse["StandardCartesianCoordinates"]
    # this returns a tuple where [0] is the atom number and atomic number and [1] is a N occurances x n atoms x 3 array
    # we want the cartesian array [1] of the last occurance [-1] to get the optimized standard cartesian coords.
    final_coords = coordies[1][-1]
    return final_coords

def calcAxes(logfile, waterCoords=None):
    coords = pullStandardCoordinates(logfile)

    if waterCoords is None:
        raise Exception("Water coordinates not defined")
    else:
        # calculate the r1 and r2 distances
        r1_vec = coords[waterCoords[0], :] - coords[waterCoords[1], :]
        r2_vec = coords[waterCoords[0], :] - coords[waterCoords[2], :]
        # calculate the three axes
        a1 = r1_vec / (np.linalg.norm(r1_vec))
        a2 = np.cross(r1_vec, r2_vec) / (np.linalg.norm(r1_vec) * np.linalg.norm(r2_vec))
        a3 = np.cross(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
    return a1, a2, a3


if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "tetramer_16", "cage")
    cageAtomStr = ["H", "H", "H", "H", "O", "O", "O", "O", "O", "O", "O", "O"]
    f1 = os.path.join(MoleculeDir, "w4c_Hw1.log")
    coords = pullStandardCoordinates(f1)
    calcAxes(coords, [0, 4, 5])
