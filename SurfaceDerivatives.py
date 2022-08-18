import os
import glob
from Converter import Constants
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
    # print(Constants.convert(ens - min(ens), "wavenumbers", to_AU=False))
    if sys_string == "w1":
        wateridx = [0, 1, 2]
    else:
        wateridx = [3, 5, 4]  # must be ordered this way because r is calculates using "vec 1"
    HOH, R12 = calcInternals(carts, wateridx)
    dipoles = allDat.Dipoles
    dataDict = {"DataName": sys_string, "MainDir": mainDir, "HOH": np.unique(HOH), "ROH": np.unique(R12),
                "Cartesians": carts, "Dipoles": dipoles, "Energies": ens}
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
    # data_name = dataDict["DataName"] + "_25pt"
    # np.save(os.path.join(dataDict["MainDir"], f"{data_name}_FDrotcoords_OHO.npy"),
    #         rot_coords)
    # np.save(os.path.join(dataDict["MainDir"], f"{data_name}_FDrotdips_OHO.npy"),
    #         dipadedodas)
    # get_xyz(os.path.join(dataDict["MainDir"], f"{data_name}_FDrotcoords_OHO.xyz"),
    #         Constants.convert(rot_coords, "angstroms", to_AU=False), dataDict["AtomStr"])
    # print("saved xyz")
    return rot_coords, dipadedodas  # bohr & debye

def spiny_spin(dataDict):
    from Eckart_turny_turn import EckartsSpinz
    from PAF_spinz import MomentOfSpinz
    massarray = np.array([Constants.mass(x) for x in ["O", "H", "H"]])
    eq_idx = np.argmin(dataDict["Energies"])
    PAobj = MomentOfSpinz(dataDict["Cartesians"][eq_idx], massarray)  # rotate eq to Principle Axis Frame
    ref = PAobj.RotCoords
    planar_flag = True
    EckObjCarts = EckartsSpinz(ref, dataDict["Cartesians"], massarray, planar=planar_flag)
    RotCoords = EckObjCarts.RotCoords  # rotate all FD steps to eq ref (PAF)
    RotDips = np.zeros_like(dataDict["Dipoles"])
    for i, dip in enumerate(dataDict["Dipoles"]):
        RotDips[i, :] = dip@EckObjCarts.TransformMat[i]  # use transformation matrix from cartesians for dipoles
    return RotCoords, RotDips

def save_DataDict(sys_string):
    dataDict1 = get_fchkData(sys_string)
    rot_coords, rot_dips = rotate(dataDict1)
    dataDict1["RotatedCoords"] = rot_coords
    dataDict1["RotatedDipoles"] = rot_dips
    data_name = dataDict1["DataName"]
    fn = os.path.join(dataDict1["MainDir"], f"{data_name}_smallDataDict.npz")
    np.savez(fn, **dataDict1)

def calcDthetaDr(dataDict):
    data_name = dataDict["DataName"]
    if os.path.exists(os.path.join(dataDict["MainDir"], f"{data_name}_smallDataDict.npz")):
        alldat = np.load(os.path.join(dataDict["MainDir"], f"{data_name}_smallDataDict.npz"), allow_pickle=True)
        dips = alldat["RotatedDipoles"]
    else:
        rot_coords, dips = rotate(dataDict)
    # calculate each dtheta dr by simple 4pt FD
    dthetadr_comps = np.zeros(3)
    for i, comp in enumerate(["X", "Y", "Z"]):
        gridDips = dips[:, i].reshape((5, 5))
        # pull needed FD points
        mm = gridDips[1, 1]
        pp = gridDips[3, 3]
        mp = gridDips[1, 3]
        pm = gridDips[3, 1]
        # calculate step sizes
        step_theta = dataDict["HOH"][2] - dataDict["HOH"][1]
        step_r = dataDict["ROH"][2] - dataDict["ROH"][1]
        # calculate dthetadr
        dthetadr_comps[i] = (mm + pp - mp - pm) / (4 * step_theta * step_r)
    dthetadr = np.linalg.norm(dthetadr_comps)
    return dthetadr

def calcDr(dataDict):
    data_name = dataDict["DataName"]
    if os.path.exists(os.path.join(dataDict["MainDir"], f"{data_name}_smallDataDict.npz")):
        alldat = np.load(os.path.join(dataDict["MainDir"], f"{data_name}_smallDataDict.npz"), allow_pickle=True)
        dips = alldat["RotatedDipoles"]
    else:
        rot_coords, dips = rotate(dataDict)
    # calculate each dtheta dr by 3pt FD @ eq theta
    dr_comps = np.zeros(3)
    for i, comp in enumerate(["X", "Y", "Z"]):
        gridDips = dips[:, i].reshape((5, 5))
        # pull needed FD points
        m = gridDips[1, 2]
        p = gridDips[3, 2]
        # calculate step size
        step_r = dataDict["ROH"][2] - dataDict["ROH"][1]
        # calculate dr
        dr_comps[i] = (p - m) / (2 * step_r)
    dr = np.linalg.norm(dr_comps)
    return dr

def calc_derivs(fd_hohs, fd_ohs, FDgrid, FDvalues):
    from McUtils.Zachary import finite_difference
    derivs = dict()
    derivs["firstHOH"] = float(finite_difference(fd_hohs, FDvalues[2, :], 1, stencil=5, only_center=True))
    derivs["firstOH"] = float(finite_difference(fd_ohs, FDvalues[:, 2], 1, stencil=5, only_center=True))
    derivs["secondHOH"] = float(finite_difference(fd_hohs, FDvalues[2, :], 2, stencil=5, only_center=True))
    derivs["secondOH"] = float(finite_difference(fd_ohs, FDvalues[:, 2], 2, stencil=5, only_center=True))
    derivs["thirdHOH"] = float(finite_difference(fd_hohs, FDvalues[2, :], 3, stencil=5, only_center=True))
    derivs["thirdOH"] = float(finite_difference(fd_ohs, FDvalues[:, 2], 3, stencil=5, only_center=True))
    derivs["mixedHOH_OH"] = float(finite_difference(FDgrid, FDvalues, (1, 1), stencil=(5, 5),
                                                    accuracy=0, only_center=True))
    derivs["mixedHOH_OHOH"] = float(finite_difference(FDgrid, FDvalues, (1, 2), stencil=(5, 5),
                                                       accuracy=0, only_center=True))
    derivs["mixedHOHHOH_OH"] = float(finite_difference(FDgrid, FDvalues, (2, 1), stencil=(5, 5),
                                                       accuracy=0, only_center=True))
    return derivs

def calc_allDerivs(dataDict):
    data_name = dataDict["DataName"]
    if os.path.exists(os.path.join(dataDict["MainDir"], f"{data_name}_smallDataDictPA.npz")):
        alldat = np.load(os.path.join(dataDict["MainDir"], f"{data_name}_smallDataDictPA.npz"), allow_pickle=True)
        dips = alldat["RotatedDipoles"]
    else:
        rot_coords, dips = rotate(dataDict)
    fd_hohs = dataDict["HOH"]
    fd_ohs = dataDict["ROH"]
    FDgrid = np.array(np.meshgrid(fd_ohs, fd_hohs)).T
    FDvaluesx = np.reshape(dips[:, 0], (5, 5))
    FDvaluesy = np.reshape(dips[:, 1], (5, 5))
    FDvaluesz = np.reshape(dips[:, 2], (5, 5))
    xderivs = calc_derivs(fd_hohs, fd_ohs, FDgrid, FDvaluesx)
    yderivs = calc_derivs(fd_hohs, fd_ohs, FDgrid, FDvaluesy)
    zderivs = calc_derivs(fd_hohs, fd_ohs, FDgrid, FDvaluesz)
    derivs = {'x': xderivs, 'y': yderivs, 'z': zderivs}
    eqDipole = np.array((FDvaluesx[2, 2], FDvaluesy[2, 2], FDvaluesz[2, 2]))
    data_name = dataDict["DataName"]
    fn = os.path.join(dataDict["MainDir"], f"{data_name}_DipCoefs.npz")
    np.savez(fn, x=xderivs, y=yderivs, z=zderivs, eqDip=eqDipole)
    return derivs

def calc_allNorms(dataDict):
    data_name = dataDict["DataName"]
    if os.path.exists(os.path.join(dataDict["MainDir"], f"{data_name}_DipCoefs.npz")):
        loadderivs = np.load(os.path.join(dataDict["MainDir"], f"{data_name}_DipCoefs.npz"), allow_pickle=True)
        derivs = {k: loadderivs[k].item() for k in ["x", "y", "z"]}
    else:
        derivs = calc_allDerivs(dataDict)
    norms = dict()
    for key in derivs["x"]:
        norms[key] = np.linalg.norm((derivs["x"][key], derivs["y"][key], derivs["z"][key]))
    fn = os.path.join(dataDict["MainDir"], f"{data_name}_DipNorms")
    np.save(fn, norms)
    return norms


if __name__ == '__main__':
    for i in ["w1"]:  # , "w2", "w6", "w6a"]:
        save_DataDict(i)
        data = get_fchkData(i)
        calc_allDerivs(data)
        a = calc_allNorms(data)
        print(a)
