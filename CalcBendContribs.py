import numpy as np
from BendAxes import calcAxes
import os

def calcWater1Contribs():
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "tetramer_16", "cage")
    cageAtomStr = ["H", "H", "H", "H", "O", "O", "O", "O", "O", "O", "O", "O"]
    f1 = os.path.join(MoleculeDir, "w4c_Hw1.log")
    w1pos = [0, 4, 5]
    a1, a2, a3 = calcAxes(f1, w1pos)
    dat = np.loadtxt(os.path.join(MoleculeDir, "Hw1_NMdisps.csv"), skiprows=1, delimiter=",")
    NMdisps = dat[:, 2:]
    for atom in w1pos:
        ...

if __name__ == '__main__':
    calcWater1Contribs()
