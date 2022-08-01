import numpy as np
import os
from BendAxes import pullStandardCoordinates
from CalcBendContribs import calcBendScaling

def calcAxis(logfile, waterPos=None):
    """ this function calculates the axis perpendicular to the HOH molecule which is used to rotate the H2O
    :return axis and origin (coordinates of the O atom)"""
    coords = pullStandardCoordinates(logfile)
    if waterPos is None:
        raise Exception("Water coordinates not defined")
    else:
        # calculate the r1 and r2 distances
        r1_vec = coords[waterPos[0], :] - coords[waterPos[1], :]
        r2_vec = coords[waterPos[0], :] - coords[waterPos[2], :]
        # calculate axis
        axis = np.cross(r1_vec, r2_vec) / (np.linalg.norm(r1_vec) * np.linalg.norm(r2_vec))
        # pull origin
        origin = coords[waterPos[0], :]
        # pull the H1 & H2 coordinates
        H1 = coords[waterPos[1], :]
        H2 = coords[waterPos[2], :]

    return axis, origin, H1, H2

def RotMat(c, t, s, X, Y, Z):
    # construct rotation Matrix for a point about an arbitrary axis
    d11 = t*X**2 + c
    d12 = t*X*Y - s*Z
    d13 = t*X*Z + s*Y
    d21 = t*X*Y + s*Z
    d22 = t*Y**2 + c
    d23 = t*Y*Z - s*X
    d31 = t*X*Z - s*Y
    d32 = t*Y*Z + s*X
    d33 = t*Z**2 + c
    rotmat = np.array([[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]])
    return rotmat

def PointRotate3D(logfile, waterPos, theta_d, angle="increase", bend_mode=None):
    """
    Return a point rotated about an arbitrary axis in 3D.
    Positive angles are counter-clockwise looking down the axis toward the origin.
    The coordinate system is assumed to be right-hand.
    Arguments: 'axis point 1', 'axis point 2', 'point to be rotated', 'angle of rotation (in degrees)' >> 'new point'
    Reference 'Rotate A Point About An Arbitrary Axis (3D)' - Paul Bourke
    Function adapted from Copyright (c) 2006 Bruce Vaughan, BV Detailing & Design, Inc.
"""
    # pull axis and origin for given water
    axis, origin, H1, H2 = calcAxis(logfile, waterPos)
    # Translate so POINTS are at origin - origin defined by the AXIS
    oH1 = H1 - origin
    oH2 = H2 - origin

    # Convert rotation axis to unit vector
    Nm = np.linalg.norm(axis)
    n = axis / Nm

    # calculate theta1 and theta2 (scaled to each H)
    SF1, SF2 = calcBendScaling(logfile, waterPos, bend_mode=bend_mode)
    print(SF1, SF2)
    theta1 = (SF1 * theta_d) * (np.pi/180)
    theta2 = (SF2 * theta_d) * (np.pi/180)
    if angle == "Decrease":
        theta2 *= -1
    elif angle == "Increase":
        theta1 *= -1
    else:
        pass

    # Matrix common factors - H1
    c = np.cos(theta1)
    t = (1 - np.cos(theta1))
    s = np.sin(theta1)
    X = n[0]
    Y = n[1]
    Z = n[2]

    M = RotMat(c, t, s, X, Y, Z)
    newH1 = np.dot(M, oH1)

    # Matrix common factors - H2
    c2 = np.cos(theta2)
    t2 = (1 - np.cos(theta2))
    s2 = np.sin(theta2)

    M2 = RotMat(c2, t2, s2, X, Y, Z)
    newH2 = np.dot(M2, oH2)

    # Translate axis and rotated point back to original location
    sH1 = newH1 + origin
    sH2 = newH2 + origin

    # calculate angle to check
    vec1 = sH1 - origin
    # print("OH1 after :", np.linalg.norm(vec1))
    vec2 = sH2 - origin
    # print("OH2 after :", np.linalg.norm(vec2))
    ang = (np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angR = np.arccos(ang)
    print(angR * (180/np.pi))

    return sH1, sH2

def writeNewCoords(logfile, waterPos, theta_d, ang_arg, atom_str, newfile, bend_mode=None, jobType="SP"):
    """ Return a written gjf file for the new geometry. """
    if bend_mode is None:
        raise Exception(f"Can not complete without bend mode defined.")
    H1, H2 = PointRotate3D(logfile, waterPos, theta_d, ang_arg, bend_mode)
    coords = pullStandardCoordinates(logfile)
    new_coords = np.copy(coords)
    new_coords[waterPos[1]] = H1
    new_coords[waterPos[2]] = H2
    with open(os.path.join(MoleculeDir, newfile), "w") as gjfFile:
        gjfFile.write(f"%chk={newfile[:-4]}.chk \n")
        gjfFile.write("%nproc=28 \n")
        gjfFile.write("%mem=120GB \n")
        if jobType == "SP":
            gjfFile.write("#p mp2/aug-cc-pvdz scf=tight density=current \n \n")
        elif jobType == "Harmonic":
            gjfFile.write("#p mp2/aug-cc-pvdz scf=tight density=current freq \n \n")
        elif jobType == "Anharmonic":
            gjfFile.write("#p mp2/aug-cc-pvdz scf=tight density=current freq=(vibrot, anh, SelectAnharmonicModes) \n \n")
            # gjfFile.write("#p mp2/aug-cc-pvdz scf=tight density=current freq=(vibrot, anh) \n \n")
        else:
            raise Exception(f"Can not determine what {jobType} job is")
        gjfFile.write("one water rest D - double zeta \n \n")
        gjfFile.write("1 1 \n")

        for i, coord in enumerate(new_coords):
            if i == waterPos[1] or i == waterPos[2]:
                # adding the ":.6f" in the curly brackets tells python to print the value to 6 decimal places
                gjfFile.write(f"{atom_str[i]}     {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f} \n")
            elif atom_str[i] == "O":
                gjfFile.write(f"{atom_str[i]}     {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f} \n")
            else:
                gjfFile.write(f"{atom_str[i]}(iso=2)     {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f} \n")
        if jobType == "Anharmonic":  # if running "SelectAnharmonicModes" specify modes to include
            gjfFile.write("\n 19-30 \n")
        gjfFile.write("\n \n \n")
        gjfFile.close()

def writeNewHODCoords(logfile, waterPos, theta_d, ang_arg, atom_str, newfile, bend_mode=None, jobType="SP"):
    """ Return a written gjf file for the new HOD geometry. """
    if bend_mode is None:
        raise Exception(f"Can not complete without bend mode defined.")
    H1, H2 = PointRotate3D(logfile, waterPos, theta_d, ang_arg, bend_mode)
    coords = pullStandardCoordinates(logfile)
    new_coords = np.copy(coords)
    new_coords[waterPos[1]] = H1
    new_coords[waterPos[2]] = H2
    with open(os.path.join(MoleculeDir, newfile), "w") as gjfFile:
        gjfFile.write(f"%chk={newfile[:-4]}.chk \n")
        gjfFile.write("%nproc=28 \n")
        gjfFile.write("%mem=120GB \n")
        if jobType == "SP":
            gjfFile.write("#p mp2/aug-cc-pvdz scf=tight density=current \n \n")
        elif jobType == "Harmonic":
            gjfFile.write("#p mp2/aug-cc-pvdz scf=tight density=current freq \n \n")
        elif jobType == "Anharmonic":
            gjfFile.write("#p mp2/aug-cc-pvdz scf=tight density=current freq=(vibrot, anh, SelectAnharmonicModes) \n \n")
        else:
            raise Exception(f"Can not determine what {jobType} job is")
        gjfFile.write("one H rest D - double zeta \n \n")
        gjfFile.write("0 1 \n")

        for i, coord in enumerate(new_coords):
            if i == waterPos[1]:  # only leave first "H" in waterPos, rest assigned D
                # adding the ":.6f" in the curly brackets tells python to print the value to 6 decimal places
                gjfFile.write(f"{atom_str[i]}     {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f} \n")
            elif atom_str[i] == "O":
                gjfFile.write(f"{atom_str[i]}     {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f} \n")
            else:
                gjfFile.write(f"{atom_str[i]}(iso=2)     {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f} \n")
        if jobType == "Anharmonic":  # if running "SelectAnharmonicModes" specify modes to include
            gjfFile.write("\n 19-30 \n")
        gjfFile.write("\n \n \n")
        gjfFile.close()


if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "hexamer_pls")
    monomer = ["H", "O", "H"]
    mono_pos = [1, 0, 2]
    tet_cage = ["O", "O", "O", "O", "H", "H", "H", "H", "H", "H", "H", "H"]
    hexa = ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
    penta = ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
    hexP = ["O", "H", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
    pentaP = ["O", "H", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
    tetP = ["O", "H", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
    tet_files = ["w4c_Hw1.log", "w4c_Hw2.log", "w4c_Hw3.log", "w4c_Hw4.log"]
    hexP_files = ["w6pT1_Hw1.log", "w6pT1_Hw2.log", "w6pT1_Hw3.log", "w6pT1_Hw4.log", "w6pT1_Hw5.log"]
    pentaP_files = ["w5pR_Hw1.log", "w5pR_Hw2.log", "w5pR_Hw3.log", "w5pR_Hw4.log"]
    tetP_files = ["w4pR_Hw1.log", "w4pR_Hw2.log", "w4pR_Hw3.log"]
    tet_pos = [[0, 4, 5], [1, 6, 7], [2, 8, 10], [3, 9, 11]]
    hex_pos = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]]
    hexp_pos = [[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
    pentap_pos = [[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    tetp_pos = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]
    bendMode = 23  # 37 for hexamer+, 35 for hexamer, 28 for pentamer, 23 for tet+, 21 for tet, 7 for di, 2 for monomer
    for f, name in enumerate(hexP_files):
        print(f+1)
        log = os.path.join(MoleculeDir, name)
        angArg = "Decrease"
        pos = hexp_pos[f]
        for i, j in enumerate(np.arange(0.5, 2.5, 0.5)):
            newFf = f"{name[:-4]}_m{i}.gjf"
            writeNewCoords(log, pos, j, angArg, hexP, newFf, bend_mode=bendMode, jobType="Harmonic")
        angArg2 = "Increase"
        for i, j in enumerate(np.arange(0.5, 2.5, 0.5)):
            newFf = f"{name[:-4]}_p{i}.gjf"
            writeNewCoords(log, pos, j, angArg2, hexP, newFf, bend_mode=bendMode, jobType="Harmonic")
