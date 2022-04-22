import numpy as np
from FChkInterpreter import FchkInterpreter
from Converter import Constants

def mass_weight(hessian, mass=None, num_coord=None, **opts):
    """Mass weights the Hessian based on given masses
        :arg hessian: full Hessian of system (get_hess)
        :arg mass: array of atomic masses (atomic units-me)
        :arg num_coord: number of atoms times 3 (3n)
        :returns gf: mass weighted hessian(ham)"""
    tot_mass = np.zeros(num_coord)
    for i, m in enumerate(mass):
        for j in range(3):
            tot_mass[i*3+j] = m
    m = 1 / np.sqrt(tot_mass)
    g = np.outer(m, m)
    gf = g*hessian
    return gf


def norms(ham):
    """solves (GF)*qn = lambda*qn
        :arg ham: Hamiltonian of system (massweighted hessian)
        :returns dictionary of frequencies squared(lambda) (atomic units) and normal mode coeficients (qns)"""
    freq2, qn = np.linalg.eigh(ham)
    normal_modes = {'freq2': freq2,
                    'qn': qn}
    return normal_modes

def calc_disps(fchkfile, water_pos):
    dat = FchkInterpreter(fchkfile)
    fchkmass = dat.atomicmasses
    mass_array = np.zeros(len(fchkmass))
    for idx, val in enumerate(fchkmass):
        if val > 15:
            mass_array[idx] = Constants.convert(val, "amu", to_AU=True)
        else:
            if idx == water_pos[1] or idx == water_pos[2]:
                mass_array[idx] = Constants.convert(val, "amu", to_AU=True)
            else:
                mass_array[idx] = Constants.mass("D", to_AU=True)
    ham = mass_weight(dat.hessian, mass_array, num_coord=(3*len(mass_array)))
    nm = norms(ham)
    disps = nm["qn"].T.reshape((3*len(mass_array)), 3, 3)
    # check shape of this lines up to shape of NMdisps from CalcBendContribs and that values are looking similar to the
    # NM disps in gaussian...
    return disps
