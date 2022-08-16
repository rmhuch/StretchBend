import numpy as np

class TM2Dexpansion:
    @classmethod
    def cubic_DM(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_hoh = params["delta_hoh"]
        delta_Roh = params["delta_oh"]
        cubic_mus = np.zeros((*delta_Roh.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            cubic_mus[:, i] = eqDip[i] + derivs[v[i]]["firstHOH"]*delta_hoh + derivs[v[i]]["firstOH"]*delta_Roh + \
                           derivs[v[i]]["secondHOH"]*(delta_hoh**2)*(1/2) + \
                           derivs[v[i]]["secondOH"]*(delta_Roh**2)*(1/2) + \
                           derivs[v[i]]["mixedHOH_OH"]*delta_Roh*delta_hoh + \
                           derivs[v[i]]["mixedHOHHOH_OH"]*(delta_hoh**2)*(1/2)*delta_Roh + \
                           derivs[v[i]]["mixedHOH_OHOH"]*delta_hoh*(delta_Roh**2)*(1/2) + \
                           derivs[v[i]]["thirdHOH"]*(delta_hoh**3)*(1/6) + \
                           derivs[v[i]]["thirdOH"]*(delta_Roh**3)*(1/6)
        return cubic_mus

    @classmethod
    def quad_DM(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_hoh = params["delta_hoh"]
        delta_Roh = params["delta_oh"]
        quad_mus = np.zeros((*delta_Roh.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            quad_mus[:, i] = eqDip[i] + derivs[v[i]]["firstHOH"]*delta_hoh + derivs[v[i]]["firstOH"]*delta_Roh + \
                           derivs[v[i]]["secondHOH"]*(delta_hoh**2)*(1/2) + \
                           derivs[v[i]]["secondOH"]*(delta_Roh**2)*(1/2) + \
                           derivs[v[i]]["mixedHOH_OH"]*delta_Roh*delta_hoh
        return quad_mus

    @classmethod
    def quadBILIN_DM(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_hoh = params["delta_hoh"]
        delta_Roh = params["delta_oh"]
        biquad_mus = np.zeros((*delta_Roh.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            biquad_mus[:, i] = eqDip[i] + derivs[v[i]]["firstHOH"]*delta_hoh + derivs[v[i]]["firstOH"]*delta_Roh + \
                           derivs[v[i]]["mixedHOH_OH"]*delta_Roh*delta_hoh
        return biquad_mus

    @classmethod
    def quadOH_DM(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_Roh = params["delta_oh"]
        ohquad_mus = np.zeros((*delta_Roh.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            ohquad_mus[:, i] = eqDip[i] + derivs[v[i]]["firstOH"]*delta_Roh + \
                               derivs[v[i]]["secondOH"]*(delta_Roh**2)*(1/2)
        return ohquad_mus

    @classmethod
    def lin_DM(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_hoh = params["delta_hoh"]
        delta_Roh = params["delta_oh"]
        lin_mus = np.zeros((*delta_Roh.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            lin_mus[:, i] = eqDip[i] + derivs[v[i]]["firstHOH"]*delta_hoh + derivs[v[i]]["firstOH"]*delta_Roh
        return lin_mus
