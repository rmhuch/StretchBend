import glob
import numpy as np
import os
from FChkInterpreter import FchkInterpreter
from McUtils.Zachary import finite_difference
from Converter import Constants

def calc_Gphiphi(r12, r23, phi123):
    # hard coded to only calculate for water..
    m1 = Constants.convert(Constants.masses["H"][0], "amu", to_AU=True)
    m2 = Constants.convert(Constants.masses["O"][0], "amu", to_AU=True)
    m3 = Constants.convert(Constants.masses["H"][0], "amu", to_AU=True)
    term1 = 1 / (m1 * r12**2)
    term2 = 1 / (m3 * r23**2)
    term3 = (1 / m2) * ((1 / r12**2) + (1 / r23**2) - (2 * np.cos(phi123) / r12 * r23))
    return term1 + term2 + term3

def calc_PotentialDerivatives(fchks, water_idx):
    dat = FchkInterpreter(*fchks)
    # pull cartesians and calculate bend angles
    carts = dat.cartesians
    HOH = []
    for geom in carts:
        vec1 = geom[water_idx[0]] - geom[water_idx[1]]
        vec2 = geom[water_idx[0]] - geom[water_idx[2]]
        ang = (np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angR = np.arccos(ang)
        HOH.append(angR)  # * (180/np.pi)
    HOH = np.array(HOH)
    # pull potential energies
    ens = dat.MP2Energy
    # order angles/energies
    sort_idx = np.argsort(HOH)
    HOH_sort = HOH[sort_idx]
    ens_sort = ens[sort_idx]
    deriv = finite_difference(HOH_sort, ens_sort, 2, stencil=5, only_center=True)
    # derivative in hartree/radian^2
    return deriv

def calc_HarmFrequency(fchks, water_idx, eqfchk):
    # calc derivative of potential
    deriv = calc_PotentialDerivatives(fchks, water_idx)
    # pull coordinates to calculate bond lengths and angles
    eqDat = FchkInterpreter(eqfchk)
    coords = eqDat.cartesians
    r12 = np.linalg.norm(coords[water_idx[0]] - coords[water_idx[1]])
    r23 = np.linalg.norm(coords[water_idx[0]] - coords[water_idx[2]])
    ang = (np.dot(r12, r23)) / (np.linalg.norm(r12) * np.linalg.norm(r23))
    angR = np.arccos(ang)
    # calculate g-matrix element
    Gphiphi = calc_Gphiphi(r12, r23, angR)
    # put it all together
    freq = np.sqrt(deriv * Gphiphi)
    return Constants.convert(freq, "wavenumbers", to_AU=False)

def calc_DipoleDerivatives(fchks, water_idx):
    dat = FchkInterpreter(*fchks)
    # pull cartesians and calculate bend angles
    carts = dat.cartesians
    HOH = []
    for geom in carts:
        vec1 = geom[water_idx[0]] - geom[water_idx[1]]
        vec2 = geom[water_idx[0]] - geom[water_idx[2]]
        ang = (np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angR = np.arccos(ang)
        HOH.append(angR)  # * (180/np.pi)
    HOH = np.array(HOH)
    # pull Dipoles
    dips = dat.Dipoles
    # order angles/energies
    sort_idx = np.argsort(HOH)
    HOH_sort = HOH[sort_idx]
    dips_sort = dips[sort_idx]
    derivs = np.zeros(3)
    for i, val in enumerate(["X", "Y", "Z"]):
        derivs[i] = finite_difference(HOH_sort, dips_sort[:, i], 1, stencil=5, only_center=True)
        # units dipole units from fchk/radians
    return Constants.convert(derivs, "debye", to_AU=False)


if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "tetramer_16", "cage")
    FDdir = os.path.join(MoleculeDir, "Water1")
    files = [os.path.join(FDdir, "w4c_Hw1.fchk"), os.path.join(FDdir, "w4c_Hw1_m0.fchk"),
             os.path.join(FDdir, "w4c_Hw1_m1.fchk"), os.path.join(FDdir, "w4c_Hw1_p0.fchk"),
             os.path.join(FDdir, "w4c_Hw1_p1.fchk")]
    # print(calc_PotentialDerivatives(files, [0, 4, 5]))
    # print(calc_DipoleDerivatives(files, [0, 4, 5]))
    print(calc_HarmFrequency(files, [0, 4, 5], os.path.join(FDdir, "w4c_Hw1.fchk")))

