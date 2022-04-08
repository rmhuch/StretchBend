import os
import numpy as np
from Converter import Constants
from FChkInterpreter import FchkInterpreter

class BuildTetCage:
    def __init__(self, isotopologue=None, FDBstep=None):
        if isotopologue is None:
            raise Exception("No isotopologue defined, can not build cluster.")
        self.isotopologue = isotopologue
        self.FDBstep = FDBstep
        self._ClusterDir = None  # Directory with specific cluster data
        self._wateridx = None  # python index of which molecules are the H2O (vs D2O)
        self._WaterDir = None  # Directory with data for specific H2O - 3 D2O isotopologue
        self._atomarray = None  # list of Atom names in the order of the Gaussian input/output
        self._massarray = None  # list of corresponding masses
        self._eqfchk = None  # path/filename of the data for the equilibrium structure (fchk of anharmonic calc)
        self._EQcartesians = None  # all cartesian coordinates at the equilibrium for given isotopologue
        self._FDBdat = None  # dictionary of fchk data for angle scan of `self.FDBstep` degrees (5 pts)
        self._waterIntCoords = None  # dictionary of water internal coordinates (from eq fchk)

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self._ClusterDir = os.path.join(docs, "stretch_bend", "tetramer_16", "cage")
        return self._ClusterDir

    @property
    def wateridx(self):
        if self._wateridx is None:
            self._wateridx = self.pullWaterIdx()
        return self._wateridx
    
    @property
    def WaterDir(self):
        if self._WaterDir is None:
            if self.wateridx is None:
                self._WaterDir = None
            else:
                waterNum = self.wateridx[0] + 1
                self._WaterDir = os.path.join(self.ClusterDir, f"Water{waterNum}")
        return self._WaterDir
    
    @property
    def atomarray(self):
        if self._atomarray is None:
            self._atomarray = self.getAtoms()
        return self._atomarray
    
    @property
    def massarray(self):
        """Uses the `self.atomarray` (list of strings) to identify atoms, pulls masses from `Constants` class and 
         converts to atomic units
        :return: masses of atoms in particular cluster
        :rtype: list of floats
        """
        if self._massarray is None:
            mass_array = []
            for A in self.atomarray:
                mass_array.append(Constants.convert(Constants.masses[A][0], Constants.masses[A][1], to_AU=True))
            self._massarray = mass_array
        return self._massarray

    @property
    def eqfchk(self):
        if self._eqfchk is None:
            waterNum = self.wateridx[0] + 1
            self._eqfchk = os.path.join(self.WaterDir, f"w4c_Hw{waterNum}.fchk")
        return self._eqfchk

    @property
    def EQcartesians(self):
        if self._EQcartesians is None:
            self._EQcartesians = self.getCarts()
        return self._EQcartesians

    @property
    def FDBdat(self):
        if self._FDBdat is None:
            self._FDBdat = self.getFDdat(step=self.FDBstep, units="degrees")
        return self._FDBdat

    @property
    def waterIntCoords(self):
        if self._waterIntCoords is None:
            self._waterIntCoords = self.calcInternals()
        return self._waterIntCoords
               
    def pullWaterIdx(self):
        """ If the isotopologue type is 1 H2O, 3 D2O this sets the `wateridx` property to the appropriately
         python indexed list ordered [O, H, H] to be used throughout the class and analysis code
         :return: the indices of the H2O in a given istopologue
         :rtype: list of ints
         """
        if self.isotopologue == "Hw1":
            wateridx = [0, 4, 5]
        elif self.isotopologue == "Hw2":
            wateridx = [1, 6, 7]
        elif self.isotopologue == "Hw3":
            wateridx = [2, 8, 10]
        elif self.isotopologue == "Hw4":
            wateridx = [3, 9, 11]
        else:
            wateridx = None
        return wateridx
            
    def getAtoms(self):
        """ for ALL isotopologues of the water tetramer cage the atom ordering is all 4 oxygens then all 8 hydrogens...
         so we start with a list of "O" then go through case by case to set the hydrogen/deuteriums
        :return: atoms of the tetramer cage ordered as they are in the Gaussian Job Files (gjf)
        :rtype: list of strings
        """
        atomarray = ["O", "O", "O", "O"]
        if self.isotopologue == "allH":
            atomarray.append(["H", "H", "H", "H", "H", "H", "H", "H"])
        else:
            atomarray.append(["D", "D", "D", "D", "D", "D", "D", "D"])
        # now go through and assign H based of isotopologue number (for 1 H cases) OR wateridx (1 H2O 3 D2O cases)
        if self.wateridx is None:
            if type(self.isotopologue) == int:
                atomarray[self.isotopologue] = "H"
            else:
                raise Exception(f"can not define atom array for {self.isotopologue} isotopologue")
        else:
            atomarray[self.wateridx[1]] = "H"
            atomarray[self.wateridx[2]] = "H"
        return atomarray

    def getCarts(self):
        """
        finds the Equilibrium Fchk file and saves the cartesian coordinates
        :return: Standard Cartesian Coordinates
        :rtype: np.array
        """
        eqDat = FchkInterpreter(self.eqfchk)
        carts = eqDat.cartesians
        return carts

    def getFDdat(self, step=None, units=None):
        waterNum = self.wateridx[0] + 1
        if units == "degrees":
            if step == "0.5":
                files = [os.path.join(self.WaterDir, f"w4c_Hw{waterNum}_m1.fchk"),
                         os.path.join(self.WaterDir, f"w4c_Hw{waterNum}_m0.fchk"),
                         os.path.join(self.WaterDir, f"w4c_Hw{waterNum}.fchk"),
                         os.path.join(self.WaterDir, f"w4c_Hw{waterNum}_p0.fchk"),
                         os.path.join(self.WaterDir, f"w4c_Hw{waterNum}_p1.fchk")]
            elif step == "1":
                files = [os.path.join(self.WaterDir, f"w4c_Hw{waterNum}_m3.fchk"),
                         os.path.join(self.WaterDir, f"w4c_Hw{waterNum}_m1.fchk"),
                         os.path.join(self.WaterDir, f"w4c_Hw{waterNum}.fchk"),
                         os.path.join(self.WaterDir, f"w4c_Hw{waterNum}_p1.fchk"),
                         os.path.join(self.WaterDir, f"w4c_Hw{waterNum}_p3.fchk")]
            else:
                raise Exception(f"Can find data with {units} units and {step} step size.")
        else:
            raise Exception(f"Can find data with {units} units")
        dat = FchkInterpreter(*files)
        # pull cartesians and calculate bend angles
        carts = dat.cartesians
        HOH = []
        for geom in carts:
            vec1 = geom[self.wateridx[0]] - geom[self.wateridx[1]]
            vec2 = geom[self.wateridx[0]] - geom[self.wateridx[2]]
            ang = (np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angR = np.arccos(ang)
            HOH.append(angR)  # * (180/np.pi)
        HOH = np.array(HOH)
        # pull potential energies
        ens = dat.MP2Energy
        # pull dipole moments
        dips = dat.Dipoles
        # order angles/energies - triple check
        sort_idx = np.argsort(HOH)
        HOH_sort = HOH[sort_idx]
        carts_sort = carts[sort_idx, :, :]
        ens_sort = ens[sort_idx]
        dips_sort = dips[sort_idx]
        data_dict = {"Cartesians": carts_sort, "HOH Angles": HOH_sort, "Energies": ens_sort, "Dipoles": dips_sort}
        return data_dict

    def calcInternals(self):
        coords = self.EQcartesians
        vec12 = coords[self.wateridx[1]] - coords[self.wateridx[0]]
        vec23 = coords[self.wateridx[2]] - coords[self.wateridx[0]]
        r12 = np.linalg.norm(vec12)
        r23 = np.linalg.norm(vec23)
        ang = (np.dot(vec12, vec23)) / (r12 * r23)
        angR = np.arccos(ang)
        data_dict = {"R12": r12, "R23": r23, "HOH": angR}
        return data_dict

    def writeFDxyz(self, file_name):
        """writes an xyz file to visualize structures from a scan.
        :arg file_name: string name of the xyz file to be written
        :returns saves an xyz file of file_name """
        crds = self.FDBdat["Cartesians"]
        atom_str = self.atomarray
        with open(file_name, 'w') as f:
            for i in range(len(crds)):
                f.write("%s \n structure %s \n" % (len(atom_str), (i+1)))
                for j in range(len(atom_str)):
                    f.write("%s %5.8f %5.8f %5.8f \n" %
                            (atom_str[j], crds[i, j, 0], crds[i, j, 1], crds[i, j, 2]))
                f.write("\n")

