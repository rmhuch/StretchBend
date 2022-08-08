import os
import numpy as np
from McUtils.GaussianInterface import GaussianLogReader

def pullZMat(logfile):
    """
    This function reads a log file and pulls out the "Standard Coordinates" (optimized Cartesian Coordinates)
    :return: array of x y z coordinates for each atom
    :rtype: array
    """
    with GaussianLogReader(logfile) as reader:
        parse = reader.parse("ZMatrices")
    coordies = parse["ZMatrices"]
    # this returns a tuple where [0] is the atom number and atomic number and [1] is a N occurances x n atoms x 3 array
    # we want the cartesian array [1] of the last occurance [-1] to get the optimized z-matrix coords.
    atom_str = coordies[0][1]
    pos_str = coordies[1]
    final_coords = coordies[2][-1]  # final z-matrix values in order
    return atom_str, pos_str, final_coords

def writeFDgjf(logfile, sys_tag, x_steps, y_steps, x_pos, y_pos):
    """ this function takes an opt log file, parses the eq geom, and then creates files in small (given steps)
    around the equilibrium and uses those geometries in SP calculations"""
    atom_str, pos_str, coords = pullZMat(logfile)
    for i, Xval in enumerate(x_steps):
        for j, Yval in enumerate(y_steps):
            new_coords = np.copy(coords)
            newfile = f"{sys_tag}_FD_X{i}_Y{j}"
            new_coords[x_pos] = new_coords[x_pos]+Xval  # here X is the OH bond length
            for pos in y_pos:         # and Y is the Bend Angle (it is defined by 1/2 so have to do twice)
                new_coords[pos] = new_coords[pos]+Yval
            with open(os.path.join(MoleculeDir, f"{newfile}.gjf"), "w") as gjfFile:
                gjfFile.write(f"%chk={newfile}.chk \n")
                gjfFile.write("%nproc=28 \n")
                gjfFile.write("%mem=120GB \n")
                gjfFile.write("#p mp2/aug-cc-pvtz scf=tight density=current \n \n")
                gjfFile.write(f"FD SP - triple zeta \n \n")
                gjfFile.write("0 1 \n")
                for k, coord in enumerate(new_coords):
                    if k == 0:
                        gjfFile.write(f"{atom_str[k]} \n")
                    elif k == 1:
                        # adding the ":.6f" in the curly brackets tells python to print the value to 6 decimal places
                        gjfFile.write(f"{atom_str[k]}  {pos_str[k, 0]}  {coord[0]:.6f}  \n")
                    elif k == 2:
                        gjfFile.write(f"{atom_str[k]}  {pos_str[k, 0]}  {coord[0]:.6f}  {pos_str[k, 1]}  {coord[1]:.6f} \n")
                    else:
                        gjfFile.write(f"{atom_str[k]}  {pos_str[k, 0]}  {coord[0]:.6f}  {pos_str[k, 1]}  {coord[1]:.6f}  {pos_str[k, 2]}  {coord[2]:.6f} \n")
                gjfFile.write("\n \n \n")
                gjfFile.close()

if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "RyanOctomer")
    logfile = os.path.join(MoleculeDir, "w6_Opt.log")
    writeFDgjf(logfile, "w6", np.linspace(-0.0125, 0.0125, num=5, endpoint=True),
               np.linspace(-1, 1, num=5, endpoint=True), (9, 0), [(8, 1), (9, 1)])
