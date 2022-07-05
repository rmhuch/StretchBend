import os
import glob
from Converter import Constants
from McUtils.Zachary import finite_difference
from Rotator import *

# helper function to calculate internal coords
def calcInternals(coords, wateridx):
    r12 = []
    HOH = []
    for geom in coords:
        vec12 = geom[wateridx[1]] - geom[wateridx[0]]
        vec23 = geom[wateridx[2]] - geom[wateridx[0]]
        r12.append(np.linalg.norm(vec12))
        ang = (np.dot(vec12, vec23)) / (np.linalg.norm(vec12) * np.linalg.norm(vec23))
        HOH.append(np.arccos(ang))
    HOH_array = np.array(HOH)
    r12_array = np.array(r12)
    return np.round(HOH_array, 5), np.round(r12_array, 5)

# locate and read in fchk files
def get_fchkData(sys_string):
    from FChkInterpreter import FchkInterpreter
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    mainDir = os.path.join(docs, "stretch_bend", "RyanOctomer")
    fchk_files = glob.glob(os.path.join(mainDir, f"{sys_string}_FD*.fchk"))
    allDat = FchkInterpreter(*fchk_files)
    carts = allDat.cartesians
    ens = allDat.MP2Energy
    # wateridx = [3, 5, 4]  # calls to donor water in all "Octomer" structures
    wateridx = [0, 1, 2]
    # ordered this way so that "r12" is the shared proton
    HOH, R12 = calcInternals(carts, wateridx)
    inds = np.lexsort((HOH, R12))
    print(HOH[inds], R12[inds])
    min_ens = ens - min(ens)
    print(Constants.convert(min_ens[inds], "wavenumbers", to_AU=False))
    dipoles = allDat.Dipoles
    dataDict = {"DataName": sys_string, "MainDir": mainDir, "HOH": np.unique(HOH), "ROH": np.unique(R12),
                "Cartesians": carts, "Dipoles": dipoles}  # PUT INDS SORT BACK IN
    return dataDict

# pull and rotate coordinates and dipoles
def rotate(dataDict):
    if dataDict["DataName"] == "w1":
        centralO_atom = 0
        xAxis_atom = 1
        xyPlane_atom = 2
        inversion_atom = None
    else:
        centralO_atom = 3  # Donor Oxygen
        xAxis_atom = 0  # Acceptor Oxygen
        inversion_atom = 5  # Shared Proton
        xyPlane_atom = None
    all_coords = Constants.convert(dataDict["Cartesians"], "angstroms", to_AU=True)
    all_dips = dataDict["Dipoles"].reshape((len(all_coords), 1, 3))
    # shift to origin
    o_coords = all_coords - all_coords[:, np.newaxis, centralO_atom]
    o_dips = all_dips - all_coords[:, np.newaxis, centralO_atom]
    # rotation to x-axis
    r1_coords, r1_dips = rot1(o_coords, o_dips, xAxis_atom)
    if dataDict["DataName"].find("w2") == 0:
        # for dimer, rotated atom is planar so "inverter" makes entire z-axis 0..
        rot_coords = r1_coords
        rot_dips = r1_dips
    elif dataDict["DataName"].find("w1") == 0:
        # for monomer, rotate the third H to the xy-plane
        rot_coords, rot_dips = rot2(r1_coords, r1_dips, xyPlane_atom, outerO1=0, outerO2=1)
    else:
        rot_coords, rot_dips = inverter(r1_coords, r1_dips, inversion_atom)  # inversion of designated atom
    dipadedodas = rot_dips.reshape(len(all_coords), 3)
    data_name = dataDict["DataName"]
    np.save(os.path.join(dataDict["MainDir"], f"{data_name}_FDrotcoords_OHO.npy"),
            rot_coords)
    np.save(os.path.join(dataDict["MainDir"], f"{data_name}_FDrotdips_OHO.npy"),
            dipadedodas)
    # get_xyz(os.path.join(dataDict["MainDir"], f"{data_name}_FDrotcoords_OHO.xyz"),
    #         Constants.convert(rot_coords, "angstroms", to_AU=False), dataDict["AtomStr"])
    # print("saved xyz")
    return rot_coords, dipadedodas  # bohr & debye

# use all 9 points to calculate mixed derivative
def calcDipDeriv(dataDict):
    data_name = dataDict["DataName"]
    # if os.path.exists(os.path.join(dataDict["MainDir"], f"{data_name}_rotdips_OHO.npy")):
    #     dips = np.load(os.path.join(dataDict["MainDir"], f"{data_name}_rotdips_OHO.npy"))
    # else:
    rot_coords, dips = rotate(dataDict)
    # calculate each dr
    dthetadr = np.zeros(3)  # return the x, y, and z dipole derivative
    for i, comp in enumerate(["X", "Y", "Z"]):
        dtheta = np.zeros(3)
        for step in np.arange(3):
            if step == 0:
                dipadodas = dips[0:3, i]
            elif step == 1:
                dipadodas = dips[3:6, i]
            elif step == 2:
                dipadodas = dips[6:, i]
            dtheta[i] = finite_difference(dataDict["HOH"], dipadodas, 1, stencil=3, only_center=True)
        # calculate dthetadr from dthetas
        dthetadr[i] = finite_difference(dataDict["ROH"], dtheta, 1, stencil=3, only_center=True)
    return dthetadr

if __name__ == '__main__':
    DD = get_fchkData("w1")
    dtdr = calcDipDeriv(DD)
    print(dtdr)

