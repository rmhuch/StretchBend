import numpy as np

def calc_EQgeom(dataDict, water_idx):
    min_arg = np.argmin(dataDict["Energies"])
    eqCarts = dataDict["Cartesians"][min_arg]
    vec12 = eqCarts[water_idx[1]] - eqCarts[water_idx[0]]
    vec23 = eqCarts[water_idx[2]] - eqCarts[water_idx[0]]
    r12 = np.linalg.norm(vec12)
    r23 = np.linalg.norm(vec23)
    ang = (np.dot(vec12, vec23)) / (r12 * r23)  # angstroms
    HOH = (np.arccos(ang))  # radians
    return r12, r23, HOH

def potlint(x=None, y=None):
    from scipy import interpolate
    tck = interpolate.splrep(x, y, s=0)

    def pf(grid, extrap=tck):
        y_fit = interpolate.splev(grid, extrap, der=0)
        return y_fit
    return pf

def make_one_Wfn_plot(grid, wfnn, idx=(0, 1)):
    import matplotlib.pyplot as plt
    colors = ["b", "r", "g", "indigo", "teal", "mediumvioletred"]
    for i in idx:
        wfnS = wfnn[:, i] * 100
        plt.plot(grid, np.repeat(0, len(grid)), "-k", linewidth=3)
        plt.plot(grid, wfnS, color=colors[i], linewidth=3)
    plt.show()

