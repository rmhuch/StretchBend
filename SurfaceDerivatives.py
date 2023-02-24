import numpy as np


def calcDthetaDr(dataDict):
    dips = dataDict["RotatedDipoles"]
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
    dips = dataDict["RotatedDipoles"]
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
    # first derivatives (2)
    derivs["firstHOH"] = float(finite_difference(fd_hohs, FDvalues[2, :], 1, stencil=5, only_center=True))
    derivs["firstOH"] = float(finite_difference(fd_ohs, FDvalues[:, 2], 1, stencil=5, only_center=True))
    # second derivatives (3)
    derivs["secondHOH"] = float(finite_difference(fd_hohs, FDvalues[2, :], 2, stencil=5, only_center=True))
    derivs["secondOH"] = float(finite_difference(fd_ohs, FDvalues[:, 2], 2, stencil=5, only_center=True))
    derivs["mixedHOH_OH"] = float(finite_difference(FDgrid, FDvalues, (1, 1), stencil=(5, 5), accuracy=0,
                                                    only_center=True))
    # third derivatives (4)
    derivs["thirdHOH"] = float(finite_difference(fd_hohs, FDvalues[2, :], 3, stencil=5, only_center=True))
    derivs["thirdOH"] = float(finite_difference(fd_ohs, FDvalues[:, 2], 3, stencil=5, only_center=True))
    derivs["mixedHOH_OHOH"] = float(finite_difference(FDgrid, FDvalues, (1, 2), stencil=(5, 5), accuracy=0,
                                                      only_center=True))
    derivs["mixedHOHHOH_OH"] = float(finite_difference(FDgrid, FDvalues, (2, 1), stencil=(5, 5), accuracy=0,
                                                       only_center=True))
    # fourth derivatives (5)
    derivs["fourthHOH"] = float(finite_difference(fd_hohs, FDvalues[2, :], 4, stencil=5, only_center=True))
    derivs["fourthOH"] = float(finite_difference(fd_ohs, FDvalues[:, 2], 4, stencil=5, only_center=True))
    derivs["mixedHOHHOHHOH_OH"] = float(finite_difference(FDgrid, FDvalues, (3, 1), stencil=(5, 5), accuracy=0,
                                                          only_center=True))
    derivs["mixedHOHHOH_OHOH"] = float(finite_difference(FDgrid, FDvalues, (2, 2), stencil=(5, 5), accuracy=0,
                                                         only_center=True))
    derivs["mixedHOH_OHOHOH"] = float(finite_difference(FDgrid, FDvalues, (1, 3), stencil=(5, 5), accuracy=0,
                                                        only_center=True))
    return derivs


def calc_allDerivs(dataDict):
    fd_hohs = dataDict["HOH"]
    fd_ohs = dataDict["ROH"]
    dips = dataDict["RotatedDipoles"]
    FDgrid = np.array(np.meshgrid(fd_ohs, fd_hohs)).T
    FDvaluesx = np.reshape(dips[:, 0], (5, 5))
    FDvaluesy = np.reshape(dips[:, 1], (5, 5))
    FDvaluesz = np.reshape(dips[:, 2], (5, 5))
    xderivs = calc_derivs(fd_hohs, fd_ohs, FDgrid, FDvaluesx)
    yderivs = calc_derivs(fd_hohs, fd_ohs, FDgrid, FDvaluesy)
    zderivs = calc_derivs(fd_hohs, fd_ohs, FDgrid, FDvaluesz)
    eqDipole = np.array((FDvaluesx[2, 2], FDvaluesy[2, 2], FDvaluesz[2, 2]))
    derivs = {'x': xderivs, 'y': yderivs, 'z': zderivs, 'eqDipole': eqDipole}
    return derivs


def calc_PotDerivs(dataDict):
    import matplotlib.pyplot as plt
    fd_hohs = dataDict["HOH"]
    fd_ohs = dataDict["ROH"]
    pot = dataDict["Energies"]
    # plt.contourf(fd_hohs, fd_ohs, pot.reshape(len(fd_ohs), len(fd_hohs)), levels=15)
    # plt.show()
    FDgrid = np.array(np.meshgrid(fd_ohs, fd_hohs)).T
    FDvalues = np.reshape(pot, (5, 5))
    derivs = calc_derivs(fd_hohs, fd_ohs, FDgrid, FDvalues)
    return derivs


def calc_allNorms(derivDict):
    norms = dict()
    for key in derivDict["x"]:
        norms[key] = np.linalg.norm((derivDict["x"][key], derivDict["y"][key], derivDict["z"][key]))
    return norms
