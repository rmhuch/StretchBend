import numpy as np


class EckartsSpinz:

    def __init__(self, reference, coords, masses, planar=None):
        '''

        :param reference: A reference structure of the given molecule
        :type reference: np array (Natoms, 3)
        :param coords: The coordinates we want to shift to the Eckart frame
        :type coords: np array (Nstructures, Natoms, 3)
        :param masses: The masses of each of the Natoms
        :type masses: np array (Natoms)
        :param planar: A flag to pass if the molecule can potentially be planar
        :type planar: True or None
        '''
        self.reference = reference
        self.coords = coords
        self.masses = masses
        self.planar = planar
        self.little_fs = np.zeros((len(coords), 3, 3))
        self.biggo_fs = np.zeros((len(coords), 3, 3))
        self.f_vecs = np.zeros((len(self.coords), 3, 3))
        self._missing_ind = None
        self._RotCoords = None
        self._TransformMat = None

    @property
    def RotCoords(self):
        if self._RotCoords is None:
            self._RotCoords = self.get_rotated_coords()
        return self._RotCoords

    @property
    def TransformMat(self):
        if self._TransformMat is None:
            self._TransformMat = self.return_f_vecs()
        return self._TransformMat

    def com_calc(self):
        '''
        This calculates the center of mass of both the reference structure and the coordinates of our walkers and shifts
        by that com
        '''
        com = np.dot(self.masses, self.coords)/np.sum(self.masses)
        self.coords = self.coords - com[:, None, :]
        ref_com = np.dot(self.masses, self.reference)/np.sum(self.masses)
        self.reference = self.reference - ref_com

    def create_f_vector_bois(self):
        '''
        This calculates the F vectors from equations 3.x of Louck and Galbraith: Eckart vectors, Eckart Frames, and
        polyatomic molecules
        '''
        mass_weight_ref = self.masses[:, None]*self.reference
        self.little_fs = np.matmul(np.transpose(self.coords, (0, 2, 1)), mass_weight_ref)
        # generates the F vectors from equation 3.1
        self._indz = np.where(np.around(self.little_fs, 4).any(axis=1))[1][:3]
        # This is a check to make sure we are or aren't planar
        self._missing_ind = np.setdiff1d(np.arange(3), self._indz)

        if self.planar is not None:
            if len(self._missing_ind) < 1:
                print("this bad boy isn't planar according to my algorithm. Please supply a reference geometry that "
                      "is on a 2d plane please")
                raise ValueError
            self._indz = self._indz[:2]
            self.little_fs = self.little_fs[:, :, self._indz]
        else:
            if len(self._missing_ind) > 0:
                print("This bad boy is a planar structure. Please pass the planar flag so the algorithm "
                      "can work properly")
                raise ValueError
        self.biggo_fs = np.matmul(self.little_fs.transpose((0, 2, 1)), self.little_fs)  # calculates the F matrix
        # from equations 3.4b and 3.4e

    def get_eigs(self):
        '''
        This obtains the eigenvalues and eigenvectors of the F (Gram) matrix
        '''
        self.create_f_vector_bois()
        self._eigs, self._eigvs = np.linalg.eigh(self.biggo_fs)  # diagonalizes the F matrix

    def get_transformed_fs(self):
        '''
        Calculation of the f unit vectors that act as the transformation vectors for our coordinates into the
        Eckart frame seen in equations 3.4a and 3.4d
        '''
        self.com_calc()
        self.get_eigs()
        eig_1o2 = 1/np.sqrt(self._eigs)[:, None, :]
        eigvsT = np.transpose(self._eigvs, (0, 2, 1))
        big_F_m1o2 = (eig_1o2*self._eigvs)@eigvsT  # calculates F^(-1/2) through a similarity transform
                                                   # used in equations 3.4a and 3.4c to get our f unit vectors
        if self.planar is None:
            self.f_vecs = np.matmul(self.little_fs, big_F_m1o2)
            mas = np.where(np.around(np.linalg.det(self.f_vecs)) == -1)
            if len(mas[0]) != 0:
                print("well, something's wrong")
                raise ValueError
        else:
            self.f_vecs[:, :, self._indz] = np.matmul(self.little_fs, big_F_m1o2)
            if self._missing_ind[0] == 1:  # f_3 is equal to f_z cross f_x
                self.f_vecs[:, :, self._missing_ind[0]] = np.cross(self.f_vecs[:, :, self._indz[1]],
                                                                    self.f_vecs[:, :, self._indz[0]])
            else:  # this is the more general formula to obtain f_3
                self.f_vecs[:, :, self._missing_ind[0]] = np.cross(self.f_vecs[:, :, self._indz[0]],
                                                                    self.f_vecs[:, :, self._indz[1]])
            mas = np.where(np.around(np.linalg.det(self.f_vecs)) == -1)
            if len(mas[0]) != 0:
                print("well, something's wrong")
                raise ValueError

    def get_rotated_coords(self):
        '''
        Use this function if you want to obtain the already transformed coordinates so you don't have to do it
        yourself
        :return: The rotated coordinates of your molecules into the Eckart frame
        :rtype: np array
        '''
        self.get_transformed_fs()
        # transform = self.f_vecs@np.transpose(self.coords, (0, 2, 1))
        return self.coords@self.f_vecs
        # return transform.transpose((0, 2, 1))

    def return_f_vecs(self):
        '''
        This is for the special people out there that either want to do the matrix multilation of the f vectors and
        their coordinates themselves, or for those that are just curious to see what the f vectors look like I guess
        :return: The f unit vectors that can rotate your coordinates into the Eckart frame
        :rtype: np array
        '''
        self.get_transformed_fs()
        return self.f_vecs


