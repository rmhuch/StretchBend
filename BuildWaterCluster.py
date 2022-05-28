import os
import numpy as np
from Converter import Constants
from FChkInterpreter import FchkInterpreter

class BuildWaterCluster:
    def __init__(self, num_waters=None, isotopologue=None, FDBstep=None):
        if num_waters > 1 and isotopologue is None:
            raise Exception("No isotopologue defined, can not build cluster.")
        self.num_waters = num_waters
        self.isotopologue = isotopologue
        self.FDBstep = FDBstep
        self._MainDir = None  # Main Directory for the project

    @property
    def MainDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._MainDir is None:
            docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self._MainDir = os.path.join(docs, "stretch_bend")
        return self._MainDir

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

    @staticmethod
    def getCarts(eqfchk):
        """
        finds the Equilibrium Fchk file and saves the cartesian coordinates
        :return: Standard Cartesian Coordinates
        :rtype: np.array
        """
        eqDat = FchkInterpreter(eqfchk)
        carts = eqDat.cartesians
        return carts

    def getFDdat(self, fchk_files, wateridx):
        dat = FchkInterpreter(*fchk_files)
        # pull cartesians and calculate internals
        carts = dat.cartesians
        HOH, R12, R23 = self.calcInternals(carts, wateridx)
        # pull potential energies
        ens = dat.MP2Energy  # (5,) - one/step
        # pull dipole moments
        dips = dat.Dipoles  # (5, 6, 3) - XYZ for 6 atoms / step
        dd = dat.DipoleDerivatives
        dd = dd.reshape((5, self.num_waters*3, 3, 3))  # XYZ for XYZ for 6 atoms / step
        # order angles/energies - triple check
        sort_idx = np.argsort(HOH)
        HOH_sort = HOH[sort_idx]
        R12_sort = R12[sort_idx]
        R23_sort = R23[sort_idx]
        carts_sort = carts[sort_idx, :, :]
        ens_sort = ens[sort_idx]
        dips_sort = dips[sort_idx]
        dd_sort = dd[sort_idx, :, :]
        data_dict = {"Cartesians": np.array(carts_sort), "R12": R12_sort, "R23": R23_sort, "HOH Angles": HOH_sort,
                     "Energies": ens_sort, "Dipoles": np.array(dips_sort), "Dipole Derivatives": np.array(dd_sort)}
        return data_dict

    def spiny_spin(self, FDBdat, massarray):
        from Eckart_turny_turn import EckartsSpinz
        from PAF_spinz import MomentOfSpinz
        RotDict = dict()
        PAobj = MomentOfSpinz(FDBdat["Cartesians"][2], massarray)  # rotate eq to Principle Axis Frame
        ref = PAobj.RotCoords
        if self.num_waters == 1:
            planar_flag = True
        else:
            planar_flag = None
        EckObjCarts = EckartsSpinz(ref, FDBdat["Cartesians"], massarray, planar=planar_flag)
        RotDict["RotCartesians"] = EckObjCarts.RotCoords  # rotate all FD steps to eq ref (PAF)
        RotDips = np.zeros_like(FDBdat["Dipoles"])
        for i, dip in enumerate(FDBdat["Dipoles"]):
            RotDips[i, :] = dip@EckObjCarts.TransformMat[i]  # use transformation matrix from cartesians for dipoles
        RotDict["RotDipoles"] = RotDips
        RotDipDerivs = np.zeros_like((FDBdat["Dipole Derivatives"]))
        for i, step in enumerate(FDBdat["Dipole Derivatives"]):  # rotate dipole derivatives
            rot1 = np.tensordot(step, EckObjCarts.TransformMat[i], axes=[1, 0])  # first rotate by cartesian (x, y, z)
            rot2 = np.tensordot(rot1, EckObjCarts.TransformMat[i], axes=[1, 0])  # then rotate dipole (x, y, z)
            RotDipDerivs[i] = rot2
        RotDict["RotDipoleDerivatives"] = RotDipDerivs
        return RotDict

    def calc_dXdR_disps(self, RotDict, wateridx, massarray):
        from Eckart_turny_turn import EckartsSpinz
        if self.num_waters == 1:
            planar_flag = True
        else:
            planar_flag = None
        DispsDict = dict()
        R1_idx = wateridx[1]
        R2_idx = wateridx[2]
        delta = Constants.convert(0.004, "angstroms", to_AU=True)
        DispsDict["delta"] = delta
        R1plus = np.copy(RotDict["RotCartesians"])
        R2plus = np.copy(RotDict["RotCartesians"])
        R1minus = np.copy(RotDict["RotCartesians"])
        R2minus = np.copy(RotDict["RotCartesians"])
        RotR1plus = np.zeros_like(R1plus)
        RotR2plus = np.zeros_like(R2plus)
        RotR1minus = np.zeros_like(R1minus)
        RotR2minus = np.zeros_like(R2minus)
        for i, step in enumerate(RotDict["RotCartesians"]):
            rOH1 = step[R1_idx, :] - step[wateridx[0], :]
            rOH2 = step[R2_idx, :] - step[wateridx[0], :]
            # calculate "R plus" coordinates (extend stretch)
            r1plus = step[R1_idx, :] + ((rOH1 / np.linalg.norm(rOH1)) * delta)
            r2plus = step[R2_idx, :] + ((rOH2 / np.linalg.norm(rOH2)) * delta)
            R1plus[i, R1_idx, :] = r1plus
            R2plus[i, R2_idx, :] = r2plus
            # calculate "R minus" coordinates (extend stretch)
            r1minus = step[R1_idx, :] - ((rOH1 / np.linalg.norm(rOH1)) * delta)
            r2minus = step[R2_idx, :] - ((rOH2 / np.linalg.norm(rOH2)) * delta)
            R1minus[i, R1_idx, :] = r1minus
            R2minus[i, R2_idx, :] = r2minus
            EckObjCarts1 = EckartsSpinz(step, R1plus[(i, ), ], massarray, planar=planar_flag)
            RotR1plus[i] = EckObjCarts1.RotCoords  # rotate the R1 plus displacements
            EckObjCarts2 = EckartsSpinz(step, R2plus[(i, ), ], massarray, planar=planar_flag)
            RotR2plus[i] = EckObjCarts2.RotCoords  # rotate the R2 plus displacements
            EckObjCarts3 = EckartsSpinz(step, R1minus[(i, ), ], massarray, planar=planar_flag)
            RotR1minus[i] = EckObjCarts3.RotCoords  # rotate the R1 minus displacements
            EckObjCarts4 = EckartsSpinz(step, R2minus[(i, ), ], massarray, planar=planar_flag)
            RotR2minus[i] = EckObjCarts4.RotCoords  # rotate the R2 minus displacements
        DispsDict["RotR1plus"] = RotR1plus
        DispsDict["RotR2plus"] = RotR2plus
        DispsDict["RotR1minus"] = RotR1minus
        DispsDict["RotR2minus"] = RotR2minus
        return DispsDict

    @staticmethod
    def calcInternals(coords, wateridx):
        if len(coords.shape) == 3:
            r12 = []
            r23 = []
            HOH = []
            for geom in coords:
                vec12 = geom[wateridx[1]] - geom[wateridx[0]]
                vec23 = geom[wateridx[2]] - geom[wateridx[0]]
                r12.append(np.linalg.norm(vec12))
                r23.append(np.linalg.norm(vec23))
                ang = (np.dot(vec12, vec23)) / (np.linalg.norm(vec12) * np.linalg.norm(vec23))
                HOH.append(np.arccos(ang))
            HOH_array = np.array(HOH)
            r12_array = np.array(r12)
            r23_array = np.array(r23)
        else:
            vec12 = coords[wateridx[1]] - coords[wateridx[0]]
            vec23 = coords[wateridx[2]] - coords[wateridx[0]]
            r12_array = np.linalg.norm(vec12)
            r23_array = np.linalg.norm(vec23)
            ang = (np.dot(vec12, vec23)) / (np.linalg.norm(vec12) * np.linalg.norm(vec23))
            HOH_array = (np.arccos(ang))
        return HOH_array, r12_array, r23_array

    @staticmethod
    def pullHarmFreqs(logfile):
        from NMParser import pull_block, format_freqs, format_disps
        dat = pull_block(logfile)
        freqs = format_freqs(dat)
        disps = format_disps(dat)
        return freqs, disps

    @staticmethod
    def pullVPT2Freqs(logfile):
        from NMParser import pull_block, pull_VPTblock, format_disps
        dat = pull_block(logfile)
        harm, vpt = pull_VPTblock(logfile)
        freqs = vpt[:, 0]
        disps = format_disps(dat)
        return freqs, disps

    @staticmethod
    def writeFDxyz(file_name, FDBdat, atomarray, WaterDir):
        """writes an xyz file to visualize structures from a scan.
        :param file_name: string name of the xyz file to be written
        :param FDBdat: dictionary of FD data for all steps
        :param atomarray: list of string atom names
        :param WaterDir: location for the file to be written to
        :returns saves an xyz file of file_name """
        crds = FDBdat["Cartesians"]
        atom_str = atomarray
        with open(os.path.join(WaterDir, file_name), 'w') as f:
            for i in range(len(crds)):
                f.write("%s \n structure %s \n" % (len(atom_str), (i+1)))
                for j in range(len(atom_str)):
                    f.write("%s %5.8f %5.8f %5.8f \n" %
                            (atom_str[j], crds[i, j, 0], crds[i, j, 1], crds[i, j, 2]))
                f.write("\n")

class BuildMonomer:
    def __init__(self):
        self.num_waters = 1
        self._ClusterDir = None  # Directory with specific cluster data
        self._atomarray = None  # list of Atom names in the order of the Gaussian input/output
        self._wateridx = None  # re-order atoms to be O, H, H (like the others) * for analysis *
        self._massarray = None  # list of corresponding masses
        self._Fchkdat = None  # FchkInterpreter Object
        self._eqlog = None
        self._waterIntCoords = None
        self._HarmFreqs = None  # harmonic Frequencies from eq point
        self._HarmDisps = None
        self._FDBdat = None  # dictionary of fchk data for angle scan of `self.FDBstep` degrees (5 pts)

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self._ClusterDir = os.path.join(docs, "stretch_bend", "monomer")
        return self._ClusterDir

    @property
    def atomarray(self):
        if self._atomarray is None:
            self._atomarray = ["H", "O", "H"]
        return self._atomarray

    @property
    def wateridx(self):
        if self._wateridx is None:
            self._wateridx = [1, 0, 2]
        return self._wateridx

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
            self._massarray = np.array(mass_array)
        return self._massarray

    @property
    def Fchkdat(self):
        if self._Fchkdat is None:
            self._Fchkdat = FchkInterpreter(os.path.join(self.ClusterDir, "monomerF3.fchk"))
        return self._Fchkdat

    @property
    def eqlog(self):
        if self._eqlog is None:
            self._eqlog = os.path.join(self.ClusterDir, "monomerF.log")
        return self._eqlog

    @property
    def waterIntCoords(self):
        if self._waterIntCoords is None:
            self._waterIntCoords = self.calcInternals()
        return self._waterIntCoords

    @property
    def HarmFreqs(self):
        if self._HarmFreqs is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs()
        return self._HarmFreqs

    @property
    def HarmDisps(self):
        if self._HarmDisps is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs()
        return self._HarmDisps

    @property
    def FDBdat(self):
        if self._FDBdat is None:
            self._FDBdat = self.getFDdat()
            self.spiny_spin()
            self.calc_dXdR_disps()
        return self._FDBdat

    def calcInternals(self):
        coords = self.Fchkdat.cartesians
        vec12 = coords[0] - coords[1]
        vec23 = coords[2] - coords[1]
        r12 = np.linalg.norm(vec12)
        r23 = np.linalg.norm(vec23)
        ang = (np.dot(vec12, vec23)) / (r12 * r23)
        angR = np.arccos(ang)
        data_dict = {"R12": r12, "R23": r23, "HOH": angR}
        return data_dict

    def pullHarmFreqs(self):
        from NMParser import pull_block, format_freqs, format_disps
        dat = pull_block(self.eqlog)
        freqs = format_freqs(dat)
        disps = format_disps(dat)
        return freqs, disps

    def getFDdat(self):
        files = [os.path.join(self.ClusterDir, f"monomer_m1_anh2.fchk"),
                 os.path.join(self.ClusterDir, f"monomer_m0_anh2.fchk"),
                 os.path.join(self.ClusterDir, f"monomerF3.fchk"),
                 os.path.join(self.ClusterDir, f"monomer_p0_anh2.fchk"),
                 os.path.join(self.ClusterDir, f"monomer_p1_anh2.fchk")]
        # files = [os.path.join(self.ClusterDir, f"HOH_m1_anh.fchk"),
        #          os.path.join(self.ClusterDir, f"HOH_m0_anh.fchk"),
        #          os.path.join(self.ClusterDir, f"monomerF3.fchk"),
        #          os.path.join(self.ClusterDir, f"HOH_p0_anh.fchk"),
        #          os.path.join(self.ClusterDir, f"HOH_p1_anh.fchk")]
        dat = FchkInterpreter(*files)
        # pull cartesians and calculate bend angles
        carts = dat.cartesians
        HOH = []  # radians
        R12 = []  # bohr
        R23 = []  # bohr
        for geom in carts:
            vec1 = geom[0] - geom[1]
            R12.append(np.linalg.norm(vec1))
            vec2 = geom[2] - geom[1]
            R23.append(np.linalg.norm(vec2))
            ang = (np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angR = np.arccos(ang)
            HOH.append(angR)  # * (180/np.pi)
        HOH = np.array(HOH)  # (5,) - one/step
        R12 = np.array(R12)  # (5,) - one/step
        R23 = np.array(R23)  # (5,) - one/step
        # pull potential energies
        ens = dat.MP2Energy  # (5,) - one/step
        # pull dipole moments
        dips = dat.Dipoles  # (5, 3, 3) - XYZ for 3 atoms / step
        dd = dat.DipoleDerivatives
        dd = dd.reshape((len(files), 3, 3, 3))  # XYZ for XYZ for 3 atoms / step
        # order angles/energies - triple check
        sort_idx = np.argsort(HOH)
        HOH_sort = HOH[sort_idx]
        R12_sort = R12[sort_idx]
        R23_sort = R23[sort_idx]
        carts_sort = carts[sort_idx, :, :]
        ens_sort = ens[sort_idx]
        dips_sort = dips[sort_idx]
        dd_sort = dd[sort_idx, :, :]
        data_dict = {"Cartesians": np.array(carts_sort), "R12": R12_sort, "R23": R23_sort, "HOH Angles": HOH_sort,
                     "Energies": ens_sort, "Dipoles": np.array(dips_sort), "Dipole Derivatives": np.array(dd_sort)}
        return data_dict

    def spiny_spin(self):
        from Eckart_turny_turn import EckartsSpinz
        from PAF_spinz import MomentOfSpinz
        PAobj = MomentOfSpinz(self.FDBdat["Cartesians"][2], self.massarray)  # rotate eq to Principle Axis Frame
        ref = PAobj.RotCoords
        EckObjCarts = EckartsSpinz(ref, self.FDBdat["Cartesians"], self.massarray, planar=True)
        self.FDBdat["RotCartesians"] = EckObjCarts.RotCoords  # rotate all FD steps to eq ref (PAF)
        RotDips = np.zeros_like(self.FDBdat["Dipoles"])
        for i, dip in enumerate(self.FDBdat["Dipoles"]):
            RotDips[i, :] = dip@EckObjCarts.TransformMat[i]  # use transformation matrix from cartesians for dipoles
        self.FDBdat["RotDipoles"] = RotDips
        RotDipDerivs = np.zeros_like((self.FDBdat["Dipole Derivatives"]))
        for i, step in enumerate(self.FDBdat["Dipole Derivatives"]):  # rotate dipole derivatives
            # dU = step.reshape((3, 3, 3))
            rot1 = np.tensordot(step, EckObjCarts.TransformMat[i], axes=[1, 1])  # first rotate by cartesian (x, y, z)
            rot2 = np.tensordot(rot1, EckObjCarts.TransformMat[i], axes=[1, 1])  # then rotate dipole (x, y, z)
            RotDipDerivs[i] = rot2  #.reshape((3, 9))  # reshape to same shape as original derivatives
        self.FDBdat["RotDipoleDerivatives"] = RotDipDerivs

    def calc_dXdR_disps(self):
        from Eckart_turny_turn import EckartsSpinz
        R1_idx = 0
        R2_idx = 2
        delta = Constants.convert(0.004, "angstroms", to_AU=True)
        self.FDBdat["delta"] = delta
        R1plus = np.copy(self.FDBdat["RotCartesians"])
        R2plus = np.copy(self.FDBdat["RotCartesians"])
        R1minus = np.copy(self.FDBdat["RotCartesians"])
        R2minus = np.copy(self.FDBdat["RotCartesians"])
        RotR1plus = np.zeros_like(R1plus)
        RotR2plus = np.zeros_like(R2plus)
        RotR1minus = np.zeros_like(R1minus)
        RotR2minus = np.zeros_like(R2minus)
        for i, step in enumerate(self.FDBdat["RotCartesians"]):
            rOH1 = step[R1_idx, :] - step[1, :]
            rOH2 = step[R2_idx, :] - step[1, :]
        # calculate "R plus" coordinates (extend stretch)
            r1plus = step[R1_idx, :] + ((rOH1 / np.linalg.norm(rOH1)) * delta)
            r2plus = step[R2_idx, :] + ((rOH2 / np.linalg.norm(rOH2)) * delta)
            R1plus[i, R1_idx, :] = r1plus
            R2plus[i, R2_idx, :] = r2plus
        # calculate "R minus" coordinates (extend stretch)
            r1minus = step[R1_idx, :] - ((rOH1 / np.linalg.norm(rOH1)) * delta)
            r2minus = step[R2_idx, :] - ((rOH2 / np.linalg.norm(rOH2)) * delta)
            R1minus[i, R1_idx, :] = r1minus
            R2minus[i, R2_idx, :] = r2minus
            EckObjCarts1 = EckartsSpinz(step, R1plus[(i, ), ], self.massarray, planar=True)
            RotR1plus[i] = EckObjCarts1.RotCoords  # rotate the R1 plus displacements
            EckObjCarts2 = EckartsSpinz(step, R2plus[(i, ), ], self.massarray, planar=True)
            RotR2plus[i] = EckObjCarts2.RotCoords  # rotate the R2 plus displacements
            EckObjCarts3 = EckartsSpinz(step, R1minus[(i, ), ], self.massarray, planar=True)
            RotR1minus[i] = EckObjCarts3.RotCoords  # rotate the R1 minus displacements
            EckObjCarts4 = EckartsSpinz(step, R2minus[(i, ), ], self.massarray, planar=True)
            RotR2minus[i] = EckObjCarts4.RotCoords  # rotate the R2 minus displacements
        self.FDBdat["RotR1plus"] = RotR1plus
        self.FDBdat["RotR2plus"] = RotR2plus
        self.FDBdat["RotR1minus"] = RotR1minus
        self.FDBdat["RotR2minus"] = RotR2minus

class BuildDimer(BuildWaterCluster):
    def __init__(self, num_waters=None, isotopologue=None, FDBstep=None):
        super().__init__(num_waters, isotopologue, FDBstep)
        self.waterNum = self.isotopologue[-1]
        self._ClusterDir = None  # Directory with specific cluster data - all data here
        self._wateridx = None  # python index of which molecules are the H2O (vs D2O) (listed O, H, H)
        self._WaterDir = None  # Directory with data for specific 1 H2O - 1 D2O isotopologue
        self._atomarray = None  # list of Atom names in the order of the Gaussian input/output
        self._massarray = None  # list of corresponding masses
        self._eqfchk = None  # path/filename of the data for the equilibrium structure (fchk of anharmonic calc)
        self._eqlog = None  # path/filename of the data for the equilibrium structure (log of anharmonic calc)
        self._EQcartesians = None  # all cartesian coordinates at the equilibrium for given isotopologue
        self._FDfiles = None  # list of the fchk files for the 5pt FD scan
        self._FDBdat = None  # dictionary of fchk data for angle scan of `self.FDBstep` degrees (5 pts)
        self._waterIntCoords = None  # dictionary of water internal coordinates (from eq fchk)
        self._HarmFreqs = None  # array of harmonic frequencies (from eq log)
        self._HarmDisps = None  # Harmonic displacements for each atom corresponding to ^ frequencies

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            self._ClusterDir = os.path.join(self.MainDir, "dimer_dz")
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
                self._WaterDir = os.path.join(self.ClusterDir, f"Water{self.waterNum}")
        return self._WaterDir

    @property
    def atomarray(self):
        if self._atomarray is None:
            self._atomarray = self.getAtoms()
        return self._atomarray

    @property
    def massarray(self):
        if self._massarray is None:
            self._massarray = self.get_masses(self.atomarray)
        return self._massarray

    @property
    def eqfchk(self):
        if self._eqfchk is None:
            self._eqfchk = os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}.fchk")
        return self._eqfchk

    @property
    def eqlog(self):
        if self._eqlog is None:
            self._eqlog = os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}.log")
        return self._eqlog

    @property
    def EQcartesians(self):
        if self._EQcartesians is None:
            self._EQcartesians = self.getCarts(self.eqfchk)
        return self._EQcartesians

    @property
    def FDfiles(self):
        if self._FDfiles is None:
            if self.FDBstep == "0.5":
                self._FDfiles = [os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}_m1_anh.fchk"),
                                 os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}_m0_anh.fchk"),
                                 os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}.fchk"),
                                 os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}_p0_anh.fchk"),
                                 os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}_p1_anh.fchk")]
            elif self.FDBstep == "1":
                self._FDfiles = [os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}_m3_anh.fchk"),
                                 os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}_m1_anh.fchk"),
                                 os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}.fchk"),
                                 os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}_p1_anh.fchk"),
                                 os.path.join(self.WaterDir, f"w2_Hw{self.waterNum}_p3_anh.fchk")]
            else:
                raise Exception(f"Can find data with {self.FDBstep} step size.")
        return self._FDfiles

    @property
    def FDBdat(self):
        if self._FDBdat is None:
            self._FDBdat = dict()
            OGFDdat = self.getFDdat(fchk_files=self.FDfiles, wateridx=self.wateridx)
            self._FDBdat.update(OGFDdat)
            RotDict = self.spiny_spin(FDBdat=OGFDdat, massarray=self.massarray)
            self._FDBdat.update(RotDict)
            DispsDict = self.calc_dXdR_disps(RotDict=RotDict, wateridx=self.wateridx, massarray=self.massarray)
            self._FDBdat.update(DispsDict)
        return self._FDBdat

    @property
    def waterIntCoords(self):
        if self._waterIntCoords is None:
            HOH, R12, R23 = self.calcInternals(coords=self.EQcartesians, wateridx=self.wateridx)
            self._waterIntCoords = {"HOH": HOH, "R12": R12, "R23": R23}
        return self._waterIntCoords

    @property
    def HarmFreqs(self):
        if self._HarmFreqs is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmFreqs

    @property
    def HarmDisps(self):
        if self._HarmDisps is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmDisps

    def pullWaterIdx(self):
        """ If the isotopologue type is 1 H2O, 1 D2O this sets the `wateridx` property to the appropriately
         python indexed list ordered [O, H, H] to be used throughout the class and analysis code
         :return: the indices of the H2O in a given istopologue
         :rtype: list of ints
         """
        if self.isotopologue == "Hw1":
            wateridx = [0, 1, 2]
        elif self.isotopologue == "Hw2":
            wateridx = [3, 4, 5]
        else:
            wateridx = None
        return wateridx

    def getAtoms(self):
        """ for ALL isotopologues of the water tetramer cage the atom ordering is all 4 oxygens then all 8 hydrogens...
         so we start with a list of "O" then go through case by case to set the hydrogen/deuteriums
        :return: atoms of the tetramer cage ordered as they are in the Gaussian Job Files (gjf)
        :rtype: list of strings
        """
        if self.isotopologue == "allH":
            atomarray = ["O", "H", "H", "O", "H", "H"]
        else:
            atomarray = ["O", "D", "D", "O", "D", "D"]
        # now go through and assign H based of isotopologue number (for 1 H cases) OR wateridx (1 H2O 1 D2O cases)
        if self.wateridx is None:
            if type(self.isotopologue) == int:
                atomarray[self.isotopologue] = "H"
            else:
                raise Exception(f"can not define atom array for {self.isotopologue} isotopologue")
        else:
            atomarray[self.wateridx[1]] = "H"
            atomarray[self.wateridx[2]] = "H"
        return atomarray

class BuildTetCage(BuildWaterCluster):
    def __init__(self, num_waters=None, isotopologue=None, FDBstep=None):
        super().__init__(num_waters, isotopologue, FDBstep)
        self.waterNum = self.isotopologue[-1]
        self._ClusterDir = None  # Directory with specific cluster data
        self._wateridx = None  # python index of which molecules are the H2O (vs D2O)
        self._WaterDir = None  # Directory with data for specific H2O - 3 D2O isotopologue
        self._atomarray = None  # list of Atom names in the order of the Gaussian input/output
        self._massarray = None  # list of corresponding masses
        self._eqfchk = None  # path/filename of the data for the equilibrium structure (fchk of anharmonic calc)
        self._eqlog = None  # path/filename of the data for the equilibrium structure (log of anharmonic calc)
        self._EQcartesians = None  # all cartesian coordinates at the equilibrium for given isotopologue
        self._FDfiles = None  # list of the fchk files for the 5pt FD scan
        self._FDBdat = None  # dictionary of fchk data for angle scan of `self.FDBstep` degrees (5 pts)
        self._waterIntCoords = None  # dictionary of water internal coordinates (from eq fchk)
        self._HarmFreqs = None  # array of harmonic frequencies (from eq log)
        self._HarmDisps = None  # Harmonic displacements for each atom corresponding to ^ frequencies

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            self._ClusterDir = os.path.join(self.MainDir, "tetramer_16", "cage")
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
                self._WaterDir = os.path.join(self.ClusterDir, f"Water{self.waterNum}")
        return self._WaterDir
    
    @property
    def atomarray(self):
        if self._atomarray is None:
            self._atomarray = self.getAtoms()
        return self._atomarray
    
    @property
    def massarray(self):
        if self._massarray is None:
            self._massarray = self.get_masses(self.atomarray)
        return self._massarray

    @property
    def eqfchk(self):
        if self._eqfchk is None:
            self._eqfchk = os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}.fchk")
        return self._eqfchk

    @property
    def eqlog(self):
        if self._eqlog is None:
            self._eqlog = os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}.log")
        return self._eqlog

    @property
    def EQcartesians(self):
        if self._EQcartesians is None:
            self._EQcartesians = self.getCarts(self.eqfchk)
        return self._EQcartesians

    @property
    def FDfiles(self):
        if self.FDBstep == "0.5":
            self._FDfiles = [os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}_m1_anh.fchk"),
                             os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}_m0_anh.fchk"),
                             os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}_p0_anh.fchk"),
                             os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}_p1_anh.fchk")]
        elif self.FDBstep == "1":
            self._FDfiles = [os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}_m1_test.fchk"),
                             os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}_m0_test.fchk"),
                             os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}_p0_test.fchk"),
                             os.path.join(self.WaterDir, f"w4c_Hw{self.waterNum}_p1_test.fchk")]
        else:
            raise Exception(f"Can find data with {self.FDBstep} step size.")
        return self._FDfiles

    @property
    def FDBdat(self):
        if self._FDBdat is None:
            self._FDBdat = dict()
            OGFDdat = self.getFDdat(fchk_files=self.FDfiles, wateridx=self.wateridx)
            self._FDBdat.update(OGFDdat)
            RotDict = self.spiny_spin(FDBdat=OGFDdat, massarray=self.massarray)
            self._FDBdat.update(RotDict)
            DispsDict = self.calc_dXdR_disps(RotDict=RotDict, wateridx=self.wateridx, massarray=self.massarray)
            self._FDBdat.update(DispsDict)
        return self._FDBdat

    @property
    def waterIntCoords(self):
        if self._waterIntCoords is None:
            HOH, R12, R23 = self.calcInternals(coords=self.EQcartesians, wateridx=self.wateridx)
            self._waterIntCoords = {"HOH": HOH, "R12": R12, "R23": R23}
        return self._waterIntCoords

    @property
    def HarmFreqs(self):
        if self._HarmFreqs is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmFreqs

    @property
    def HarmDisps(self):
        if self._HarmDisps is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmDisps
               
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
            atomarray.extend(["H", "H", "H", "H", "H", "H", "H", "H"])
        else:
            atomarray.extend(["D", "D", "D", "D", "D", "D", "D", "D"])
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

class BuildTetThreeOne(BuildWaterCluster):
    def __init__(self, num_waters=None, isotopologue=None, FDBstep=None):
        super().__init__(num_waters, isotopologue, FDBstep)
        self.waterNum = self.isotopologue[-1]
        self._ClusterDir = None  # Directory with specific cluster data
        self._wateridx = None  # python index of which molecules are the H2O (vs D2O)
        self._WaterDir = None  # Directory with data for specific H2O - 3 D2O isotopologue
        self._atomarray = None  # list of Atom names in the order of the Gaussian input/output
        self._massarray = None  # list of corresponding masses
        self._eqfchk = None  # path/filename of the data for the equilibrium structure (fchk of anharmonic calc)
        self._eqlog = None  # path/filename of the data for the equilibrium structure (log of anharmonic calc)
        self._EQcartesians = None  # all cartesian coordinates at the equilibrium for given isotopologue
        self._FDfiles = None  # list of the fchk files for the 5pt FD scan
        self._FDBdat = None  # dictionary of fchk data for angle scan of `self.FDBstep` degrees (5 pts)
        self._waterIntCoords = None  # dictionary of water internal coordinates (from eq fchk)
        self._HarmFreqs = None
        self._HarmDisps = None  # Harmonic displacements for each atom corresponding to ^ frequencies

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            self._ClusterDir = os.path.join(self.MainDir, "tetramer_16", "three_one")
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
                self._WaterDir = os.path.join(self.ClusterDir, f"Water{self.waterNum}")
        return self._WaterDir

    @property
    def atomarray(self):
        if self._atomarray is None:
            self._atomarray = self.getAtoms()
        return self._atomarray

    @property
    def massarray(self):
        if self._massarray is None:
            self._massarray = self.get_masses(self.atomarray)
        return self._massarray

    @property
    def eqfchk(self):
        if self._eqfchk is None:
            self._eqfchk = os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}.fchk")
        return self._eqfchk

    @property
    def eqlog(self):
        if self._eqlog is None:
            self._eqlog = os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}.log")
        return self._eqlog

    @property
    def EQcartesians(self):
        if self._EQcartesians is None:
            self._EQcartesians = self.getCarts(self.eqfchk)
        return self._EQcartesians

    @property
    def FDfiles(self):
        if self.FDBstep == "0.5":
            self._FDfiles = [os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}_m1.fchk"),
                             os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}_m0.fchk"),
                             os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}_p0.fchk"),
                             os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}_p1.fchk")]
        elif self.FDBstep == "1":
            self._FDfiles = [os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}_m3.fchk"),
                             os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}_m1.fchk"),
                             os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}_p1.fchk"),
                             os.path.join(self.WaterDir, f"w4t_Hw{self.waterNum}_p3.fchk")]
        else:
            raise Exception(f"Can find data with {self.FDBstep} step size.")
        return self._FDfiles

    @property
    def FDBdat(self):
        if self._FDBdat is None:
            self._FDBdat = dict()
            OGFDdat = self.getFDdat(fchk_files=self.FDfiles, wateridx=self.wateridx)
            self._FDBdat.update(OGFDdat)
            RotDict = self.spiny_spin(FDBdat=OGFDdat, massarray=self.massarray)
            self._FDBdat.update(RotDict)
            DispsDict = self.calc_dXdR_disps(RotDict=RotDict, wateridx=self.wateridx, massarray=self.massarray)
            self._FDBdat.update(DispsDict)
        return self._FDBdat

    @property
    def waterIntCoords(self):
        if self._waterIntCoords is None:
            HOH, R12, R23 = self.calcInternals(coords=self.EQcartesians, wateridx=self.wateridx)
            self._waterIntCoords = {"HOH": HOH, "R12": R12, "R23": R23}
        return self._waterIntCoords

    @property
    def HarmFreqs(self):
        if self._HarmFreqs is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmFreqs

    @property
    def HarmDisps(self):
        if self._HarmDisps is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmDisps

    def pullWaterIdx(self):
        """ If the isotopologue type is 1 H2O, 3 D2O this sets the `wateridx` property to the appropriately
         python indexed list ordered [O, H, H] to be used throughout the class and analysis code
         :return: the indices of the H2O in a given istopologue
         :rtype: list of ints
         """
        if self.isotopologue == "Hw1":
            wateridx = [0, 3, 4]
        elif self.isotopologue == "Hw2":
            wateridx = [1, 5, 6]
        elif self.isotopologue == "Hw3":
            wateridx = [2, 7, 8]
        elif self.isotopologue == "Hw4":
            wateridx = [9, 10, 11]
        else:
            wateridx = None
        return wateridx

    def getAtoms(self):
        """ for ALL isotopologues of the water tetramer three one the atom ordering is 3 oxygens then 6 hydrogens,
            to make the ring then the "one" water listed last. so we start with a list of "O" then go through case by
            case to set the hydrogen/deuteriums
        :return: atoms of the tetramer cage ordered as they are in the Gaussian Job Files (gjf)
        :rtype: list of strings
        """
        atomarray = ["O", "O", "O"]
        if self.isotopologue == "allH":
            atomarray.extend(["H", "H", "H", "H", "H", "H", "O", "H", "H"])
        else:
            atomarray.extend(["D", "D", "D", "D", "D", "D", "O", "D", "D"])
        # now go through and assign H based of isotopologue number (for 1 H cases) OR wateridx (1 H2O 3 D2O cases)
        if self.wateridx is None:
            # Not sure if this will work in all cases, will need to revisit if ever need to use
            if type(self.isotopologue) == int:
                atomarray[self.isotopologue] = "H"
            else:
                raise Exception(f"can not define atom array for {self.isotopologue} isotopologue")
        else:
            atomarray[self.wateridx[1]] = "H"
            atomarray[self.wateridx[2]] = "H"
        return atomarray

class BuildPentCage(BuildWaterCluster):
    def __init__(self, num_waters=None, isotopologue=None, FDBstep=None):
        super().__init__(num_waters, isotopologue, FDBstep)
        self.waterNum = self.isotopologue[-1]
        self._ClusterDir = None  # Directory with specific cluster data
        self._wateridx = None  # python index of which molecules are the H2O (vs D2O)
        self._WaterDir = None  # Directory with data for specific H2O - 4 D2O isotopologue
        self._atomarray = None  # list of Atom names in the order of the Gaussian input/output
        self._massarray = None  # list of corresponding masses
        self._eqfchk = None  # path/filename of the data for the equilibrium structure (fchk of anharmonic calc)
        self._eqlog = None  # path/filename of the data for the equilibrium structure (log of anharmonic calc)
        self._EQcartesians = None  # all cartesian coordinates at the equilibrium for given isotopologue
        self._FDfiles = None  # list of the fchk files for the 5pt FD scan
        self._FDBdat = None  # dictionary of fchk data for angle scan of `self.FDBstep` degrees (5 pts)
        self._waterIntCoords = None  # dictionary of water internal coordinates (from eq fchk - all H)
        self._HarmFreqs = None
        self._HarmDisps = None  # Harmonic displacements for each atom corresponding to ^ frequencies

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            self._ClusterDir = os.path.join(self.MainDir, "pentamer", "cage")
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
                self._WaterDir = os.path.join(self.ClusterDir, f"Water{self.waterNum}")
        return self._WaterDir

    @property
    def atomarray(self):
        if self._atomarray is None:
            self._atomarray = self.getAtoms()
        return self._atomarray

    @property
    def massarray(self):
        if self._massarray is None:
            self._massarray = self.get_masses(self.atomarray)
        return self._massarray

    @property
    def eqfchk(self):
        if self._eqfchk is None:
            self._eqfchk = os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}.fchk")
        return self._eqfchk

    @property
    def eqlog(self):
        if self._eqlog is None:
            self._eqlog = os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}.log")
        return self._eqlog

    @property
    def EQcartesians(self):
        if self._EQcartesians is None:
            self._EQcartesians = self.getCarts(self.eqfchk)
        return self._EQcartesians

    @property
    def FDfiles(self):
        if self.FDBstep == "0.5":
            self._FDfiles = [os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}_m1_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}_m0_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}_p0_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}_p1_harm.fchk")]
        elif self.FDBstep == "1":
            self._FDfiles = [os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}_m3_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}_m1_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}_p1_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5c_Hw{self.waterNum}_p3_harm.fchk")]
        else:
            raise Exception(f"Can find data with {self.FDBstep} step size.")
        return self._FDfiles

    @property
    def FDBdat(self):
        if self._FDBdat is None:
            self._FDBdat = dict()
            OGFDdat = self.getFDdat(fchk_files=self.FDfiles, wateridx=self.wateridx)
            self._FDBdat.update(OGFDdat)
            RotDict = self.spiny_spin(FDBdat=OGFDdat, massarray=self.massarray)
            self._FDBdat.update(RotDict)
            DispsDict = self.calc_dXdR_disps(RotDict=RotDict, wateridx=self.wateridx, massarray=self.massarray)
            self._FDBdat.update(DispsDict)
        return self._FDBdat

    @property
    def waterIntCoords(self):
        if self._waterIntCoords is None:
            HOH, R12, R23 = self.calcInternals(coords=self.EQcartesians, wateridx=self.wateridx)
            self._waterIntCoords = {"HOH": HOH, "R12": R12, "R23": R23}
        return self._waterIntCoords

    @property
    def HarmFreqs(self):
        if self._HarmFreqs is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmFreqs

    @property
    def HarmDisps(self):
        if self._HarmDisps is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmDisps

    def pullWaterIdx(self):
        """ If the isotopologue type is 1 H2O, 5 D2O this sets the `wateridx` property to the appropriately
         python indexed list ordered [O, H, H] to be used throughout the class and analysis code
         :return: the indices of the H2O in a given istopologue
         :rtype: list of ints
         """
        if self.isotopologue == "Hw1":
            wateridx = [0, 1, 2]
        elif self.isotopologue == "Hw2":
            wateridx = [3, 4, 5]
        elif self.isotopologue == "Hw3":
            wateridx = [6, 7, 8]
        elif self.isotopologue == "Hw4":
            wateridx = [9, 10, 11]
        elif self.isotopologue == "Hw5":
            wateridx = [12, 13, 14]
        else:
            wateridx = None
        return wateridx

    def getAtoms(self):
        """
        :return: atoms of the pentamer cage ordered as they are in the Gaussian Job Files (gjf)
        :rtype: list of strings
        """
        if self.isotopologue == "allH":
            atomarray = ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
        else:
            atomarray = ["O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D"]
        # now go through and assign H based of isotopologue number (for 1 H cases) OR wateridx (1 H2O 4 D2O cases)
        if self.wateridx is None:
            # Not sure if this will work in all cases, will need to revisit if ever need to use
            if type(self.isotopologue) == int:
                atomarray[self.isotopologue] = "H"
            else:
                raise Exception(f"can not define atom array for {self.isotopologue} isotopologue")
        else:
            atomarray[self.wateridx[1]] = "H"
            atomarray[self.wateridx[2]] = "H"
        return atomarray

class BuildPentRing(BuildWaterCluster):
    def __init__(self, num_waters=None, isotopologue=None, FDBstep=None):
        super().__init__(num_waters, isotopologue, FDBstep)
        self.waterNum = self.isotopologue[-1]
        self._ClusterDir = None  # Directory with specific cluster data
        self._wateridx = None  # python index of which molecules are the H2O (vs D2O)
        self._WaterDir = None  # Directory with data for specific H2O - 5 D2O isotopologue
        self._atomarray = None  # list of Atom names in the order of the Gaussian input/output
        self._massarray = None  # list of corresponding masses
        self._eqfchk = None  # path/filename of the data for the equilibrium structure (fchk of anharmonic calc)
        self._eqlog = None  # path/filename of the data for the equilibrium structure (log of anharmonic calc)
        self._EQcartesians = None  # all cartesian coordinates at the equilibrium for given isotopologue
        self._FDfiles = None  # list of the fchk files for the 5pt FD scan
        self._FDBdat = None  # dictionary of fchk data for angle scan of `self.FDBstep` degrees (5 pts)
        self._waterIntCoords = None  # dictionary of water internal coordinates (from eq fchk - all H)
        self._HarmFreqs = None  # array of harmonic frequencies (from eq log)
        self._HarmDisps = None  # Harmonic displacements for each atom corresponding to ^ frequencies

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            self._ClusterDir = os.path.join(self.MainDir, "pentamer", "ring")
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
                self._WaterDir = os.path.join(self.ClusterDir, f"Water{self.waterNum}")
        return self._WaterDir

    @property
    def atomarray(self):
        if self._atomarray is None:
            self._atomarray = self.getAtoms()
        return self._atomarray

    @property
    def massarray(self):
        if self._massarray is None:
            self._massarray = self.get_masses(self.atomarray)
        return self._massarray

    @property
    def eqfchk(self):
        if self._eqfchk is None:
            self._eqfchk = os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}.fchk")
        return self._eqfchk

    @property
    def eqlog(self):
        if self._eqlog is None:
            self._eqlog = os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}.log")
        return self._eqlog

    @property
    def EQcartesians(self):
        if self._EQcartesians is None:
            self._EQcartesians = self.getCarts(self.eqfchk)
        return self._EQcartesians

    @property
    def FDfiles(self):
        if self.FDBstep == "0.5":
            self._FDfiles = [os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}_m1_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}_m0_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}_p0_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}_p1_harm.fchk")]
        elif self.FDBstep == "1":
            self._FDfiles = [os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}_m3_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}_m1_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}_p1_harm.fchk"),
                             os.path.join(self.WaterDir, f"w5r_Hw{self.waterNum}_p3_harm.fchk")]
        else:
            raise Exception(f"Can't find data with {self.FDBstep} step size.")
        return self._FDfiles

    @property
    def FDBdat(self):
        if self._FDBdat is None:
            self._FDBdat = dict()
            OGFDdat = self.getFDdat(fchk_files=self.FDfiles, wateridx=self.wateridx)
            self._FDBdat.update(OGFDdat)
            RotDict = self.spiny_spin(FDBdat=OGFDdat, massarray=self.massarray)
            self._FDBdat.update(RotDict)
            DispsDict = self.calc_dXdR_disps(RotDict=RotDict, wateridx=self.wateridx, massarray=self.massarray)
            self._FDBdat.update(DispsDict)
        return self._FDBdat

    @property
    def waterIntCoords(self):
        if self._waterIntCoords is None:
            HOH, R12, R23 = self.calcInternals(coords=self.EQcartesians, wateridx=self.wateridx)
            self._waterIntCoords = {"HOH": HOH, "R12": R12, "R23": R23}
        return self._waterIntCoords

    @property
    def HarmFreqs(self):
        if self._HarmFreqs is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmFreqs

    @property
    def HarmDisps(self):
        if self._HarmDisps is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmDisps

    def pullWaterIdx(self):
        """ If the isotopologue type is 1 H2O, 5 D2O this sets the `wateridx` property to the appropriately
         python indexed list ordered [O, H, H] to be used throughout the class and analysis code
         :return: the indices of the H2O in a given istopologue
         :rtype: list of ints
         """
        if self.isotopologue == "Hw1":
            wateridx = [0, 1, 2]
        elif self.isotopologue == "Hw2":
            wateridx = [3, 4, 5]
        elif self.isotopologue == "Hw3":
            wateridx = [6, 7, 8]
        elif self.isotopologue == "Hw4":
            wateridx = [9, 10, 11]
        elif self.isotopologue == "Hw5":
            wateridx = [12, 13, 14]
        else:
            wateridx = None
        return wateridx

    def getAtoms(self):
        """
        :return: atoms of the pentamer ring ordered as they are in the Gaussian Job Files (gjf)
        :rtype: list of strings
        """
        if self.isotopologue == "allH":
            atomarray = ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
        else:
            atomarray = ["O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D"]
        # now go through and assign H based of isotopologue number (for 1 H cases) OR wateridx (1 H2O 5 D2O cases)
        if self.wateridx is None:
            # Not sure if this will work in all cases, will need to revisit if ever need to use
            if type(self.isotopologue) == int:
                atomarray[self.isotopologue] = "H"
            else:
                raise Exception(f"can not define atom array for {self.isotopologue} isotopologue")
        else:
            atomarray[self.wateridx[1]] = "H"
            atomarray[self.wateridx[2]] = "H"
        return atomarray

class BuildHexCage(BuildWaterCluster):
    def __init__(self, num_waters=None, isotopologue=None, FDBstep=None):
        super().__init__(num_waters, isotopologue, FDBstep)
        self.waterNum = self.isotopologue[-1]
        self._ClusterDir = None  # Directory with specific cluster data
        self._wateridx = None  # python index of which molecules are the H2O (vs D2O)
        self._WaterDir = None  # Directory with data for specific H2O - 5 D2O isotopologue
        self._atomarray = None  # list of Atom names in the order of the Gaussian input/output
        self._massarray = None  # list of corresponding masses
        self._eqfchk = None  # path/filename of the data for the equilibrium structure (fchk of anharmonic calc)
        self._eqlog = None  # path/filename of the data for the equilibrium structure (log of anharmonic calc)
        self._EQcartesians = None  # all cartesian coordinates at the equilibrium for given isotopologue
        self._FDfiles = None  # list of the fchk files for the 5pt FD scan
        self._FDBdat = None  # dictionary of fchk data for angle scan of `self.FDBstep` degrees (5 pts)
        self._waterIntCoords = None  # dictionary of water internal coordinates (from eq fchk - all H)
        self._HarmFreqs = None
        self._HarmDisps = None  # Harmonic displacements for each atom corresponding to ^ frequencies

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            self._ClusterDir = os.path.join(self.MainDir, "hexamer_dz", "cage")
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
                self._WaterDir = os.path.join(self.ClusterDir, f"Water{self.waterNum}")
        return self._WaterDir

    @property
    def atomarray(self):
        if self._atomarray is None:
            self._atomarray = self.getAtoms()
        return self._atomarray

    @property
    def massarray(self):
        if self._massarray is None:
            self._massarray = self.get_masses(self.atomarray)
        return self._massarray

    @property
    def eqfchk(self):
        if self._eqfchk is None:
            self._eqfchk = os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}.fchk")
        return self._eqfchk

    @property
    def eqlog(self):
        if self._eqlog is None:
            self._eqlog = os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}.log")
        return self._eqlog

    @property
    def EQcartesians(self):
        if self._EQcartesians is None:
            self._EQcartesians = self.getCarts(self.eqfchk)
        return self._EQcartesians

    @property
    def FDfiles(self):
        if self.FDBstep == "0.5":
            self._FDfiles = [os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}_m1_harm.fchk"),
                             os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}_m0_harm.fchk"),
                             os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}_p0_harm.fchk"),
                             os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}_p1_harm.fchk")]
        elif self.FDBstep == "1":
            self._FDfiles = [os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}_m3_harm.fchk"),
                             os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}_m1_harm.fchk"),
                             os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}_p1_harm.fchk"),
                             os.path.join(self.WaterDir, f"w6c_Hw{self.waterNum}_p3_harm.fchk")]
        else:
            raise Exception(f"Can find data with {self.FDBstep} step size.")
        return self._FDfiles

    @property
    def FDBdat(self):
        if self._FDBdat is None:
            self._FDBdat = dict()
            OGFDdat = self.getFDdat(fchk_files=self.FDfiles, wateridx=self.wateridx)
            self._FDBdat.update(OGFDdat)
            RotDict = self.spiny_spin(FDBdat=OGFDdat, massarray=self.massarray)
            self._FDBdat.update(RotDict)
            DispsDict = self.calc_dXdR_disps(RotDict=RotDict, wateridx=self.wateridx, massarray=self.massarray)
            self._FDBdat.update(DispsDict)
        return self._FDBdat

    @property
    def waterIntCoords(self):
        if self._waterIntCoords is None:
            HOH, R12, R23 = self.calcInternals(coords=self.EQcartesians, wateridx=self.wateridx)
            self._waterIntCoords = {"HOH": HOH, "R12": R12, "R23": R23}
        return self._waterIntCoords

    @property
    def HarmFreqs(self):
        if self._HarmFreqs is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmFreqs

    @property
    def HarmDisps(self):
        if self._HarmDisps is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmDisps

    def pullWaterIdx(self):
        """ If the isotopologue type is 1 H2O, 5 D2O this sets the `wateridx` property to the appropriately
         python indexed list ordered [O, H, H] to be used throughout the class and analysis code
         :return: the indices of the H2O in a given istopologue
         :rtype: list of ints
         """
        if self.isotopologue == "Hw1":
            wateridx = [0, 1, 2]
        elif self.isotopologue == "Hw2":
            wateridx = [3, 4, 5]
        elif self.isotopologue == "Hw3":
            wateridx = [6, 7, 8]
        elif self.isotopologue == "Hw4":
            wateridx = [9, 10, 11]
        elif self.isotopologue == "Hw5":
            wateridx = [12, 13, 14]
        elif self.isotopologue == "Hw6":
            wateridx = [15, 16, 17]
        else:
            wateridx = None
        return wateridx

    def getAtoms(self):
        """ for ALL isotopologues of the water tetramer three one the atom ordering is 3 oxygens then 6 hydrogens,
            to make the ring then the "one" water listed last. so we start with a list of "O" then go through case by
            case to set the hydrogen/deuteriums
        :return: atoms of the tetramer cage ordered as they are in the Gaussian Job Files (gjf)
        :rtype: list of strings
        """
        if self.isotopologue == "allH":
            atomarray = ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
        else:
            atomarray = ["O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D"]
        # now go through and assign H based of isotopologue number (for 1 H cases) OR wateridx (1 H2O 5 D2O cases)
        if self.wateridx is None:
            # Not sure if this will work in all cases, will need to revisit if ever need to use
            if type(self.isotopologue) == int:
                atomarray[self.isotopologue] = "H"
            else:
                raise Exception(f"can not define atom array for {self.isotopologue} isotopologue")
        else:
            atomarray[self.wateridx[1]] = "H"
            atomarray[self.wateridx[2]] = "H"
        return atomarray

class BuildHexPrism(BuildWaterCluster):
    def __init__(self, num_waters=None, isotopologue=None, FDBstep=None):
        super().__init__(num_waters, isotopologue, FDBstep)
        self.waterNum = self.isotopologue[-1]
        self._ClusterDir = None  # Directory with specific cluster data
        self._wateridx = None  # python index of which molecules are the H2O (vs D2O)
        self._WaterDir = None  # Directory with data for specific H2O - 5 D2O isotopologue
        self._atomarray = None  # list of Atom names in the order of the Gaussian input/output
        self._massarray = None  # list of corresponding masses
        self._eqfchk = None  # path/filename of the data for the equilibrium structure (fchk of anharmonic calc)
        self._eqlog = None  # path/filename of the data for the equilibrium structure (log of anharmonic calc)
        self._EQcartesians = None  # all cartesian coordinates at the equilibrium for given isotopologue
        self._FDfiles = None  # list of the fchk files for the 5pt FD scan
        self._FDBdat = None  # dictionary of fchk data for angle scan of `self.FDBstep` degrees (5 pts)
        self._waterIntCoords = None  # dictionary of water internal coordinates (from eq fchk - all H)
        self._HarmFreqs = None
        self._HarmDisps = None  # Harmonic displacements for each atom corresponding to ^ frequencies

    @property
    def ClusterDir(self):
        """uses os to pull the directory where all data for this project is stored"""
        if self._ClusterDir is None:
            self._ClusterDir = os.path.join(self.MainDir, "hexamer_dz", "prism")
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
                self._WaterDir = os.path.join(self.ClusterDir, f"Water{self.waterNum}")
        return self._WaterDir

    @property
    def atomarray(self):
        if self._atomarray is None:
            self._atomarray = self.getAtoms()
        return self._atomarray

    @property
    def massarray(self):
        if self._massarray is None:
            self._massarray = self.get_masses(self.atomarray)
        return self._massarray

    @property
    def eqfchk(self):
        if self._eqfchk is None:
            self._eqfchk = os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}.fchk")
        return self._eqfchk

    @property
    def eqlog(self):
        if self._eqlog is None:
            self._eqlog = os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}.log")
        return self._eqlog

    @property
    def EQcartesians(self):
        if self._EQcartesians is None:
            self._EQcartesians = self.getCarts(self.eqfchk)
        return self._EQcartesians

    @property
    def FDfiles(self):
        if self.FDBstep == "0.5":
            self._FDfiles = [os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_m1.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_m0.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_p0.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_p1.fchk")]
        elif self.FDBstep == "0.25":
            self._FDfiles = [os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_p2_TO.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_m0_TO.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_p0_TO.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_p1_TO.fchk")]
        elif self.FDBstep == "1":
            self._FDfiles = [os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_m3.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_m1.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_p1.fchk"),
                             os.path.join(self.WaterDir, f"w6p_Hw{self.waterNum}_p3.fchk")]
        else:
            raise Exception(f"Can find data with {self.FDBstep} step size.")
        return self._FDfiles

    @property
    def FDBdat(self):
        if self._FDBdat is None:
            self._FDBdat = dict()
            OGFDdat = self.getFDdat(fchk_files=self.FDfiles, wateridx=self.wateridx)
            self._FDBdat.update(OGFDdat)
            RotDict = self.spiny_spin(FDBdat=OGFDdat, massarray=self.massarray)
            self._FDBdat.update(RotDict)
            DispsDict = self.calc_dXdR_disps(RotDict=RotDict, wateridx=self.wateridx, massarray=self.massarray)
            self._FDBdat.update(DispsDict)
        return self._FDBdat

    @property
    def waterIntCoords(self):
        if self._waterIntCoords is None:
            HOH, R12, R23 = self.calcInternals(coords=self.EQcartesians, wateridx=self.wateridx)
            self._waterIntCoords = {"HOH": HOH, "R12": R12, "R23": R23}
        return self._waterIntCoords

    @property
    def HarmFreqs(self):
        if self._HarmFreqs is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmFreqs

    @property
    def HarmDisps(self):
        if self._HarmDisps is None:
            self._HarmFreqs, self._HarmDisps = self.pullHarmFreqs(self.eqlog)
        return self._HarmDisps

    def pullWaterIdx(self):
        """ If the isotopologue type is 1 H2O, 3 D2O this sets the `wateridx` property to the appropriately
         python indexed list ordered [O, H, H] to be used throughout the class and analysis code
         :return: the indices of the H2O in a given istopologue
         :rtype: list of ints
         """
        if self.isotopologue == "Hw1":
            wateridx = [0, 1, 2]
        elif self.isotopologue == "Hw2":
            wateridx = [3, 4, 5]
        elif self.isotopologue == "Hw3":
            wateridx = [6, 7, 8]
        elif self.isotopologue == "Hw4":
            wateridx = [9, 10, 11]
        elif self.isotopologue == "Hw5":
            wateridx = [12, 13, 14]
        elif self.isotopologue == "Hw6":
            wateridx = [15, 16, 17]
        else:
            wateridx = None
        return wateridx

    def getAtoms(self):
        """ for ALL isotopologues of the water tetramer three one the atom ordering is 3 oxygens then 6 hydrogens,
            to make the ring then the "one" water listed last. so we start with a list of "O" then go through case by
            case to set the hydrogen/deuteriums
        :return: atoms of the tetramer cage ordered as they are in the Gaussian Job Files (gjf)
        :rtype: list of strings
        """
        if self.isotopologue == "allH":
            atomarray = ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
        else:
            atomarray = ["O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D", "O", "D", "D"]
        # now go through and assign H based of isotopologue number (for 1 H cases) OR wateridx (1 H2O 5 D2O cases)
        if self.wateridx is None:
            # Not sure if this will work in all cases, will need to revisit if ever need to use
            if type(self.isotopologue) == int:
                atomarray[self.isotopologue] = "H"
            else:
                raise Exception(f"can not define atom array for {self.isotopologue} isotopologue")
        else:
            atomarray[self.wateridx[1]] = "H"
            atomarray[self.wateridx[2]] = "H"
        return atomarray

