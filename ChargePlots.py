import os
import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants

def plot_deltaHvsHOH(dataDict):
    HOH = dataDict["xyData"][:, 1] * (180/np.pi)  # pull HOH and convert to degrees
    MC = dataDict["MullikenCharges"]  # mulliken charges at every point O H(scan coord) H
    eq_idx = np.argmin(dataDict["Energies"])
    eq_coords = dataDict["xyData"][eq_idx]
    roh_eq = np.argwhere(dataDict["xyData"][:, 0] == eq_coords[0])
    MCatEQ = MC[roh_eq].squeeze()
    HOHatEQ = HOH[roh_eq].squeeze()
    DeltaH = MCatEQ[:, 1] - MC[eq_idx, 1]
    squareXY = dataDict["xyData"].reshape(31, 31, 2)
    squareMC = MC.reshape(31, 31, 3)
    for idx in np.arange(len(squareXY)):
        xy = squareXY[idx]
        charge = squareMC[idx]
        rOHang = Constants.convert(xy[0, 0], "angstroms", to_AU=False)
        if idx <= 5 or idx >= 20:
            pass
        # elif idx % 2 == 0:
        #     pass
        else:
            plt.plot(xy[5:21, 1]*(180/np.pi), charge[5:21, 1] - MC[eq_idx, 1], "o", label=f"rOH = {rOHang}")
    plt.plot(HOHatEQ[5:21], DeltaH[5:21], "o", color="k")
    plt.plot(HOHatEQ[5:21], np.repeat(0, len(HOHatEQ[5:21])))
    plt.legend(bbox_to_anchor=(0.8, 0.5), loc='center left')
    plt.show()

if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MainDir = os.path.join(docs, "stretch_bend", "RyanOctomer")
    DD = np.load(os.path.join(MainDir, "w1_RB_bigDataDictPA.npz"), allow_pickle=True)
    plot_deltaHvsHOH(DD)

