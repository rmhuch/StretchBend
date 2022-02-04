import numpy as np
import os
from Converter import Constants
from McUtils import Numputils


def pull_coordsandmodes(coord_vals, modedir):
    coords = []
    modes = []
    for i in coord_vals:
        dat = np.loadtxt(os.path.join(modedir, f"tbhp_eq_v{i}_modes.dat"))  # f"tbhp_eq_{i}oh_modes.dat"
        coordies = Constants.convert(np.reshape(dat[:1, :], (-1, 3)), "angstroms", to_AU=False)
        coords.append(coordies)
        big_mode_data = Constants.convert(dat[1:, :], "angstroms", to_AU=False)
        # modies = np.reshape(big_mode_data, (len(big_mode_data), -1, 3)) / demass[np.newaxis, :, np.newaxis]
        modes.append(big_mode_data)  # np.reshape(modies, big_mode_data.shape)
    return np.array(coords), np.array(modes)

def OOHContribs(coord_vals, modedir):
    """Returns a dictionary keyed by rOH values with a 2column array as the key.
    [:, 0]: mode numbers and [:, 1]: absolute value of contributions"""
    coords, modes = pull_coordsandmodes(coord_vals, modedir)
    contribDict = dict()
    for val, coord, mode in zip(coord_vals, coords, modes):
        di, dj, dk = Numputils.angle_deriv(coord, [5], [4], [6])
        deriv_OOH = np.zeros((len(coord), 3))
        deriv_OOH[5] = di
        deriv_OOH[4] = dj
        deriv_OOH[6] = dk
        norm_derivOOH = deriv_OOH / np.linalg.norm(deriv_OOH)
        contribs = np.dot(mode, norm_derivOOH.flatten())
        norm_contribs = (contribs / np.linalg.norm(contribs))**2
        idx = np.where(abs(norm_contribs) > 0.3)[0]
        contribdat = np.column_stack((idx, abs(norm_contribs[idx])))
        sortdat = contribdat[contribdat[:, 1].argsort()[::-1]]
        contribDict[val] = sortdat
    return contribDict
