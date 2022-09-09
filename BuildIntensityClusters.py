import os
import glob
import numpy as np
from Converter import Constants

class BuildIntensityCluster:
    def __init__(self, num_atoms=None, isotopologue=None):
        if num_atoms > 3 and isotopologue is None:
            raise Exception("No isotopologue defined, can not build cluster.")
        self.num_atoms = num_atoms
        self.isotopologue = isotopologue
        self._MainDir = None  # Main Directory for the project
        self._MainFigDir = None  # Main Figure Directory for the project

    @property
    def MainDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._MainDir is None:
            docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self._MainDir = os.path.join(docs, "stretch_bend", "RyanOctomer")
        return self._MainDir

    @property
    def MainFigDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._MainFigDir is None:
            self._MainFigDir = os.path.join(self.MainDir, "Figures")
        return self._MainFigDir

    def parse_logData(self, cluster_dir, sys_str, water_idx=None, scan_coords=None):
        """ takes a list of log files and returns a dictionary filled with Gaussian data from the logfiles
            :param cluster_dir, the directory for the exact cluster we are parsing the data for
            :param sys_str, system identifying tag
            :param water_idx, the index of the scanned water, only necessary when optimized scan
            :param scan_coords, tuple of strings of coordinate names in optimized scan, keep None if rigid
            :returns dictionary ('Dipoles' - dipoles from Gaussian (later rotated),
                                 'xyData' - values of the scanned coordinates, 'Energies' - Gaussian energies,
                                 'Cartesians' -  Coordinates at every scan point,
                                 'Mulliken Charges' - Mulliken charge of each atom at every scan point"""
        from McUtils.GaussianInterface import GaussianLogReader
        dataDict = dict()
        # parse through logfiles to collect data
        dips = []  # dipoles straight of log file
        xyData = []  # values of the scanned coordinates (a.u.)
        ens = []  # energies of each geometry (minimum shifted)
        cats = []  # cartesian coordinates of each geometry
        allCharges = []  # Mulliken charges of each atom at each geometry
        logfile_pattern = f"{sys_str}Scan*.log"
        logfiless = glob.glob(os.path.join(cluster_dir, "Scans", logfile_pattern))
        for logpath in logfiless:
            if self.isotopologue == "optimized":
                cart_struct = dict()  # start an empty ordered dict for each file
                charge_struct = dict()
                with GaussianLogReader(logpath) as reader:
                    parse = reader.parse(("OptimizedScanEnergies", "OptimizedDipoleMoments",
                                          "StandardCartesianCoordinates", "MullikenCharges"))
                raw_data = parse["OptimizedScanEnergies"]  # format energies
                ens.append(raw_data[0])  # returns MP2 energy
                xyData.append(np.column_stack((raw_data[1][scan_coords[0]], raw_data[1][scan_coords[1]])))
                dips.append(list(parse["OptimizedDipoleMoments"]))  # format dipoles
                ccs = parse["StandardCartesianCoordinates"][1]
                # since the cartesian coordinates and Mulliken charges return every step not every optimized step,
                # we have to trim the data...
                x_vals, y_vals = self.calc_scoords(ccs, water_idx)  # format cartesians - x in angstroms, y in radians
                y_deg = np.round(y_vals * (180 / np.pi), 4)
                coords = zip(x_vals, y_deg, ccs)  # TRIPLE CHECK THIS
                cart_struct.update((((x, y), cc) for x, y, cc in coords))
                cats.append(list(cart_struct.values()))
                Mcharges = (parse["MullikenCharges"])  # format Mulliken Charges
                Cdat = []
                for i in Mcharges:
                    pt = i.split()
                    Cdat.append([float(pt[k]) for k in np.arange(3, len(pt), 3)])  # pulls the charges of all the atoms
                charges = zip(x_vals, y_vals, Cdat)
                charge_struct.update((((x, y), mc) for x, y, mc in charges))
                allCharges.append(list(charge_struct.values()))
            else:
                with GaussianLogReader(logpath) as reader:
                    parse = reader.parse(("ScanEnergies", "DipoleMoments",
                                          "StandardCartesianCoordinates", "MullikenCharges"))
                raw_data = parse["ScanEnergies"][1]  # format energies
                # if logpath[-5] is "a":  # patches hole in w6 ADD/AAD scan
                #     xyData.append(np.column_stack((raw_data[:, 1], np.repeat(64.1512, len(raw_data[:, 1])))))
                # else:
                xyData.append(np.column_stack((raw_data[:, 1], raw_data[:, 2])))
                ens.append(raw_data[:, -1])  # returns MP2 energy
                dips.append(list(parse["DipoleMoments"]))  # format dipoles
                cats.append(parse["StandardCartesianCoordinates"][1])  # format cartesians
                Mcharges = (parse["MullikenCharges"])  # format Mulliken Charges
                dat = []
                for i in Mcharges:
                    pt = i.split()
                    dat.append([float(pt[k]) for k in np.arange(3, len(pt), 3)])  # pulls the charges of all the atoms
                allCharges.append(dat)
        # concatenate lists
        dipoles = np.concatenate(dips)
        scoords = np.concatenate(xyData)
        energy = np.concatenate(ens)
        energy = energy - min(energy)
        carts = np.concatenate(cats)
        MullCharges = np.concatenate(allCharges)
        # Remove the odd angles to have a square grid from rigid dimer and all monomer
        if self.isotopologue == "rigid" or self.num_atoms == 1:
            evenIdx = np.argwhere(scoords[:, 1] % 2 < 1).squeeze()
            dipoles = dipoles[evenIdx]
            scoords = scoords[evenIdx]
            energy = energy[evenIdx]
            carts = carts[evenIdx]
            MullCharges = MullCharges[evenIdx]
        else:
            pass
        # convert scan coords into a.u.
        x_au = Constants.convert(scoords[:, 0], "angstroms", to_AU=True)
        y_au = (scoords[:, 1] * 2) * (np.pi / 180)  # convert scan coords from bisector to HOH and to radians
        scoords_au = np.column_stack((x_au, y_au))
        # make sure sorted & place in dictionary
        idx = np.lexsort((scoords_au[:, 1], scoords_au[:, 0]))  # OH is slow moving coordinate, hoh is fast
        dataDict["Dipoles"] = Constants.convert(dipoles[idx], "debye", to_AU=True)
        dataDict["xyData"] = scoords_au[idx]
        dataDict["Energies"] = energy[idx]
        dataDict["Cartesians"] = Constants.convert(carts[idx], "angstroms", to_AU=True)
        dataDict["MullikenCharges"] = MullCharges[idx]
        return dataDict

    def parse_fchkData(self, cluster_dir, sys_str, water_idx):
        from FChkInterpreter import FchkInterpreter
        fchk_files = glob.glob(os.path.join(cluster_dir, "FDpts", f"{sys_str}FD*.fchk"))
        allDat = FchkInterpreter(*fchk_files)
        carts = allDat.cartesians
        ens = allDat.MP2Energy
        R12, HOH = self.calc_scoords(carts, water_idx)  # should be in bohr and radians
        dipoles = allDat.Dipoles
        idx = np.lexsort((HOH, R12))  # OH is slow moving coordinate, hoh is fast
        sort_carts = carts[idx]
        sort_dips = dipoles[idx]
        sort_ens = ens[idx]
        dataDict = {"HOH": np.unique(HOH[idx]), "ROH": np.unique(R12[idx]),
                    "Cartesians": sort_carts, "Dipoles": sort_dips, "Energies": sort_ens}
        return dataDict

    @staticmethod
    def calc_scoords(cartesians, water_idx):
        """Calculates the two OH bond lengths and the HOH angle, the OH bond lengths will be returned in whatever
        unit they come in and the HOH angle will always be returned in radians"""
        xList = []
        yList = []
        for Cart in cartesians:
            vec12 = Cart[water_idx[1]] - Cart[water_idx[0]]
            vec23 = Cart[water_idx[2]] - Cart[water_idx[0]]
            r12 = np.linalg.norm(vec12)
            r23 = np.linalg.norm(vec23)
            ang = (np.dot(vec12, vec23)) / (r12 * r23)
            HOH = (np.arccos(ang))
            xList.append(r12)
            yList.append(HOH)
        x_array = np.round(np.array(xList), 4)
        y_array = np.round(np.array(yList), 4)
        return x_array, y_array

    @staticmethod
    def spiny_spin(data_dict, atom_str):
        from Eckart_turny_turn import EckartsSpinz
        from PAF_spinz import MomentOfSpinz
        massarray = np.array([Constants.mass(x) for x in atom_str])
        eq_idx = np.argmin(data_dict["Energies"])
        PAobj = MomentOfSpinz(data_dict["Cartesians"][eq_idx], massarray)  # rotate eq to Principle Axis Frame
        ref = PAobj.RotCoords
        if len(atom_str) == 3:
            planar_flag = True
        else:
            planar_flag = None
        EckObjCarts = EckartsSpinz(ref, data_dict["Cartesians"], massarray, planar=planar_flag)
        RotCoords = EckObjCarts.RotCoords  # rotate all FD steps to eq ref (PAF)
        RotDips = np.zeros_like(data_dict["Dipoles"])
        for i, dip in enumerate(data_dict["Dipoles"]):
            RotDips[i, :] = dip @ EckObjCarts.TransformMat[i]  # use transformation matrix from cartesians for dipoles
        return RotCoords, RotDips

    def save_DataDict(self, cluster_dir, sys_str, atom_str, water_idx, dtype=None, scan_coords=None):
        if dtype is None:
            raise Exception("type of data to parse is not defined, use `dtype` keyword to define big or small scan")
        elif dtype == "big":
            dataDict1 = self.parse_logData(cluster_dir=cluster_dir, sys_str=sys_str,
                                           water_idx=water_idx, scan_coords=scan_coords)
        elif dtype == "small":
            dataDict1 = self.parse_fchkData(cluster_dir=cluster_dir, sys_str=sys_str, water_idx=water_idx)
        else:
            raise Exception(f"dtype {dtype} is not supported.")
        rot_coords, rot_dips = self.spiny_spin(data_dict=dataDict1, atom_str=atom_str)
        dataDict1["RotatedCoords"] = rot_coords
        dataDict1["RotatedDipoles"] = rot_dips
        return dataDict1

    @staticmethod
    def get_masses(atomarray):
        """Uses the `self.atomarray` (list of strings) to identify atoms, pulls masses from `Constants` class and
             converts to atomic units
            :return: masses of atoms in particular cluster
            :rtype: list of floats
            """
        mass_array = []
        for A in atomarray:
            mass_array.append(Constants.convert(Constants.masses[A][0], Constants.masses[A][1], to_AU=True))
        return np.array(mass_array)

class BuildW1(BuildIntensityCluster):
    def __init__(self, num_atoms=1, isotopologue=None):
        super().__init__(num_atoms, isotopologue)
        if isotopologue is None:
            raise Exception("use 'isotopologue' to identify monomer scan (rigid or optimized) and try again")
        self.isotopologue = isotopologue
        self.AtomStr = ["O", "H", "H"]
        self.WaterIdx = [0, 1, 2]
        self.scanCoords = None
        self._ClusterDir = None
        self._SysStr = None
        self._BigScanDataDict = None
        self._SmallScanDataDict = None

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            self._ClusterDir = os.path.join(self.MainDir, "w1")
        return self._ClusterDir

    @property
    def SysStr(self):
        """identifies the system string - the tag at the beginning of all the files used and created"""
        if self._SysStr is None:
            if self.isotopologue == "rigid":
                self._SysStr = "w1_"
            elif self.isotopologue == "optimized":
                self._SysStr = "w1_O_"
                self.scanCoords = ["R1", "A1"]
            else:
                raise Exception(f"unknown isotopolgoue: {self.isotopologue} for w1")
        return self._SysStr

    @property
    def BigScanDataDict(self):
        if self._BigScanDataDict is None:
            DDfile = os.path.join(self.ClusterDir, f"{self.SysStr}bigDataDict.npz")
            if os.path.exists(DDfile):
                self._BigScanDataDict = np.load(DDfile, allow_pickle=True)
            else:
                BSDD = self.save_DataDict(cluster_dir=self.ClusterDir, sys_str=self.SysStr, atom_str=self.AtomStr,
                                          water_idx=self.WaterIdx, dtype="big", scan_coords=self.scanCoords)
                np.savez(DDfile, **BSDD)
                print(f"saved Data to {DDfile}")
                self._BigScanDataDict = BSDD
        return self._BigScanDataDict

    @property
    def SmallScanDataDict(self):
        if self._SmallScanDataDict is None:
            DDfile = os.path.join(self.ClusterDir, f"{self.SysStr}smallDataDict.npz")
            if os.path.exists(DDfile):
                self._SmallScanDataDict = np.load(DDfile, allow_pickle=True)
            else:
                SSDD = self.save_DataDict(cluster_dir=self.ClusterDir, sys_str=self.SysStr, atom_str=self.AtomStr,
                                          water_idx=self.WaterIdx, dtype="small")
                np.savez(DDfile, **SSDD)
                print(f"Data saved to {DDfile}")
                self._SmallScanDataDict = SSDD
        return self._SmallScanDataDict

class BuildW2(BuildIntensityCluster):
    def __init__(self, num_atoms=2, isotopologue=None, Hbound=None):
        super().__init__(num_atoms, isotopologue)
        if isotopologue is None:
            raise Exception("use 'isotopologue' to identify dimer scan (rigid or optimized) and try again")
        self.isotopologue = isotopologue
        self.AtomStr = ["O", "H", "H", "O", "H", "H"]
        if Hbound is None:
            raise Exception("use 'Hbound' to identify if we are interested in a bound of free OH and try again")
        self.Hbound = Hbound
        if self.Hbound:
            self.WaterIdx = [3, 5, 4]
        else:
            self.WaterIdx = [3, 4, 5]
        self.scanCoords = None
        self._ClusterDir = None
        self._SysStr = None
        self._BigScanDataDict = None
        self._SmallScanDataDict = None

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            self._ClusterDir = os.path.join(self.MainDir, "w2")
        return self._ClusterDir

    @property
    def SysStr(self):
        """identifies the system string - the tag at the beginning of all the files used and created"""
        if self._SysStr is None:
            if self.Hbound:
                sysStr = "w2_R5B_"
                self.scanCoords = ["R5", "A2"]
            else:
                sysStr = "w2_R4B_"
                self.scanCoords = ["R4", "A2"]
            if self.isotopologue == "optimized":
                sysStr += "O_"
            else:
                pass
            self._SysStr = sysStr
        return self._SysStr

    @property
    def BigScanDataDict(self):
        if self._BigScanDataDict is None:
            DDfile = os.path.join(self.ClusterDir, f"{self.SysStr}bigDataDict.npz")
            if os.path.exists(DDfile):
                self._BigScanDataDict = np.load(DDfile, allow_pickle=True)
            else:
                BSDD = self.save_DataDict(cluster_dir=self.ClusterDir, sys_str=self.SysStr, atom_str=self.AtomStr,
                                          water_idx=self.WaterIdx, dtype="big", scan_coords=self.scanCoords)
                np.savez(DDfile, **BSDD)
                print(f"Data saved to {DDfile}")
                self._BigScanDataDict = BSDD
        return self._BigScanDataDict

    @property
    def SmallScanDataDict(self):
        if self._SmallScanDataDict is None:
            DDfile = os.path.join(self.ClusterDir, f"{self.SysStr}smallDataDict.npz")
            if os.path.exists(DDfile):
                self._SmallScanDataDict = np.load(DDfile, allow_pickle=True)
            else:
                SSDD = self.save_DataDict(cluster_dir=self.ClusterDir, sys_str=self.SysStr,
                                                             atom_str=self.AtomStr, water_idx=self.WaterIdx,
                                                             dtype="small")
                np.savez(DDfile, **SSDD)
                print(f"Data saved to {DDfile}")
                self._SmallScanDataDict = SSDD
        return self._SmallScanDataDict
