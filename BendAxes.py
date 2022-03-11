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
    """ this function calculates _two_ axes systems, each centered at one of the hydrogen atoms in a water molecule.
        the axes are defined as follows: the OH bond, perpendicular to the HOH molecule, and in-plane with the HOH"""
    coords = pullStandardCoordinates(logfile)
    if waterCoords is None:
        raise Exception("Water coordinates not defined")
    else:
        # calculate the r1 and r2 distances
        r1_vec = coords[waterCoords[0], :] - coords[waterCoords[1], :]
        r2_vec = coords[waterCoords[0], :] - coords[waterCoords[2], :]
        # calculate the three axes - for H1
        a11 = r1_vec / (np.linalg.norm(r1_vec))
        a21 = np.cross(r1_vec, r2_vec) / (np.linalg.norm(r1_vec) * np.linalg.norm(r2_vec))
        a31 = np.cross(a11, a21) / (np.linalg.norm(a11) * np.linalg.norm(a21))
        # stack results
        res1 = np.column_stack((a11, a21, a31))
        # calculate the three axes - for H2
        a12 = r2_vec / (np.linalg.norm(r2_vec))
        a22 = np.cross(r1_vec, r2_vec) / (np.linalg.norm(r1_vec) * np.linalg.norm(r2_vec))
        a32 = np.cross(a12, a22) / (np.linalg.norm(a12) * np.linalg.norm(a22))
        # stack results
        res2 = np.column_stack((a12, a22, a32))
    return res1, res2


if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "tetramer_16", "three_one")
    f1 = os.path.join(MoleculeDir, "w4t_Hw1.log")
    coords = pullStandardCoordinates(f1)
    calcAxes(coords, [0, 3, 4])
