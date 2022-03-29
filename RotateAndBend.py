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

def PointRotate3D(logfile, waterPos, theta_d, angle="increase"):
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
    # Translate so axis AND POINTS are at origin
    N = axis - origin
    oH1 = H1 - origin
    oH2 = H2 - origin

    # Rotation axis unit vector
    Nm = np.linalg.norm(N)
    n = np.array([N[0]/Nm, N[1]/Nm, N[2]/Nm])

    # calculate theta1 and theta2 (scaled to each H)
    SF1, SF2 = calcBendScaling(logfile, waterPos)
    theta1 = (SF1 * theta_d) * (np.pi/180)
    # print(theta1)
    theta2 = (SF2 * theta_d) * (np.pi/180)
    # print(theta2)
    if angle == "Increase":
        theta1 *= -1
        theta2 *= -1
    else:
        pass

    # Matrix common factors - H1
    c = np.cos(theta1)
    t = (1 - np.cos(theta1))
    s = np.sin(theta1)
    X = n[0]
    Y = n[1]
    Z = n[2]

    # construct rotation Matrix 'M'
    d11 = t*X**2 + c
    d12 = t*X*Y - s*Z
    d13 = t*X*Z + s*Y
    d21 = t*X*Y + s*Z
    d22 = t*Y**2 + c
    d23 = t*Y*Z - s*X
    d31 = t*X*Z - s*Y
    d32 = t*Y*Z + s*X
    d33 = t*Z**2 + c

    #            |rotPt.x|
    # Matrix 'M'*|rotPt.y|
    #            |rotPt.z|
    newH1 = np.zeros(3)
    newH1[0] = d11*oH1[0] + d12*oH1[0] + d13*oH1[0]
    newH1[1] = d21*oH1[1] + d22*oH1[1] + d23*oH1[1]
    newH1[2] = d31*oH1[2] + d32*oH1[2] + d33*oH1[2]

    # Matrix common factors - H2
    c2 = np.cos(theta2)
    t2 = (1 - np.cos(theta2))
    s2 = np.sin(theta2)

    # construct rotation Matrix 'M'
    d11_ = t2*X**2 + c2
    d12_ = t2*X*Y - s2*Z
    d13_ = t2*X*Z + s2*Y
    d21_ = t2*X*Y + s2*Z
    d22_ = t2*Y**2 + c2
    d23_ = t2*Y*Z - s2*X
    d31_ = t2*X*Z - s2*Y
    d32_ = t2*Y*Z + s2*X
    d33_ = t2*Z**2 + c2

    #            |rotPt.x|
    # Matrix 'M'*|rotPt.y|
    #            |rotPt.z|
    newH2 = np.zeros(3)
    newH2[0] = d11_*oH2[0] + d12_*oH2[0] + d13_*oH2[0]
    newH2[1] = d21_*oH2[1] + d22_*oH2[1] + d23_*oH2[1]
    newH2[2] = d31_*oH2[2] + d32_*oH2[2] + d33_*oH2[2]

    # Translate axis and rotated point back to original location
    sH1 = newH1 + origin
    sH2 = newH2 + origin

    # calculate angle to check
    vec1 = origin - sH1
    vec2 = origin - sH2
    ang = (np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angR = np.arccos(ang)
    print(angR * (180/np.pi))

    return sH1, sH2

def writeNewCoords(logfile, waterPos, theta_d, ang_arg, newfile, atom_str):
    """ Return a written gjf file for the new geometry. """
    H1, H2 = PointRotate3D(logfile, waterPos, theta_d, ang_arg)
    coords = pullStandardCoordinates(logfile)
    new_coords = np.copy(coords)
    new_coords[waterPos[1]] = H1
    new_coords[waterPos[2]] = H2
    with open(os.path.join(MoleculeDir, newfile), "w") as gjfFile:
        gjfFile.write(f"%chk={newfile[:-4]}.chk \n")
        gjfFile.write("%nproc=28 \n")
        gjfFile.write("%mem=120GB \n")
        gjfFile.write("#p mp2/aug-cc-pvdz scf=tight opt=verytight density=current \n \n")
        gjfFile.write("one water rest D - Tetramer cage double zeta \n \n")
        gjfFile.write("0 1 \n")

        for i, coord in enumerate(new_coords):
            if i == waterPos[1] or i == waterPos[2]:
                gjfFile.write(f"{atom_str[i]}     {coord[0]}  {coord[1]}  {coord[2]} \n")
            elif atom_str[i] == "O":
                gjfFile.write(f"{atom_str[i]}     {coord[0]}  {coord[1]}  {coord[2]} \n")
            else:
                gjfFile.write(f"{atom_str[i]}(iso=2)     {coord[0]}  {coord[1]}  {coord[2]} \n")

        gjfFile.write("\n \n \n")
        gjfFile.close()


if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "tetramer_16", "cage")
    f1 = os.path.join(MoleculeDir, "w4c_Hw1.log")
    newF = "w4c_Hw1move1_test.gjf"
    cage = ["O", "O", "O", "O", "H", "H", "H", "H", "H", "H", "H", "H"]
    ang_arg = "Decrease"
    for i, j in enumerate(np.arange(0.5, 6.5, 0.5)):
        newFf = f"w4c_Hw1_m{i}.gjf"
        writeNewCoords(f1, [0, 4, 5], j, ang_arg, newFf, cage)

