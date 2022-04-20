from McUtils.GaussianInterface import GaussianFChkReader
import numpy as np


class FchkInterpreter:
    def __init__(self, *fchks, **kwargs):
        self.params = kwargs
        if len(fchks) == 0:
            raise Exception('Nothing to interpret.')
        self.fchks = fchks
        self._hessian = None
        self._IntHessian = None
        self._cartesians = None  # dictionary of cartesian coordinates keyed by (x, y) distances
        self._gradient = None
        self._MP2Energy = None
        self._atomicmasses = None
        self._Dipoles = None
        self._DipoleDerivatives = None

    @property
    def cartesians(self):
        if self._cartesians is None:
            self._cartesians = self.get_coords()
        return self._cartesians

    @property
    def hessian(self):
        if self._hessian is None:
            self._hessian = self.get_hess()
        return self._hessian

    @property
    def IntHessian(self):
        if self._IntHessian is None:
            self._IntHessian = self.get_IntHess()
        return self._IntHessian

    @property
    def gradient(self):
        if self._gradient is None:
            self._gradient = self.get_grad()
        return self._gradient

    @property
    def MP2Energy(self):
        if self._MP2Energy is None:
            self._MP2Energy = self.get_MP2energy()
        return self._MP2Energy

    @property
    def atomicmasses(self):
        if self._atomicmasses is None:
            self._atomicmasses = self.get_mass()
        return self._atomicmasses

    @property
    def Dipoles(self):
        if self._Dipoles is None:
            self._Dipoles = self.get_DipoleMoment()
        return self._Dipoles

    @property
    def DipoleDerivatives(self):
        if self._DipoleDerivatives is None:
            self._DipoleDerivatives = self.get_DipoleDerivatives()
        return self._DipoleDerivatives

    def get_coords(self):
        """Uses McUtils parser to pull cartesian coordinates
            :returns coords: nx3 coordinate matrix"""
        crds = []
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("Coordinates")
            coords = parse["Coordinates"]
            crds.append(coords)
        c = np.array(crds)
        if c.shape[0] == 1:
            c = np.squeeze(c)
        return c

    def get_hess(self):
        """Pulls the Hessian (Force Constants) from a Gaussian Frequency output file
            :arg fchk_file: a Gaussian Frequency formatted checkpoint file
            :returns hess: full Hessian of system as an np.array"""
        forcies = []
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("ForceConstants")
                forceC = parse["ForceConstants"]  # Gaussian "Force Constants" is just lower triangle of Hessian Matrix
            forcies.append(forceC.array)  # returns the full Hessian
        f = np.array(forcies)
        if f.shape[0] == 1:
            f = np.squeeze(f)
        return f

    def get_IntHess(self):
        """Pulls the Internal Coordinate Hessian (Force Constants) from a Gaussian Frequency output file
            :arg fchk_file: a Gaussian Frequency formatted checkpoint file
            :returns hess: full Hessian of system as an np.array"""
        forcies = []
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("InternalForceConstants")
                forceC = parse["InternalForceConstants"]
                # Gaussian "Force Constants" is just lower triangle of Hessian Matrix
            forcies.append(forceC.array)  # returns the full Hessian
        f = np.array(forcies)
        if f.shape[0] == 1:
            f = np.squeeze(f)
        return f

    def get_grad(self):
        grad = []
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("Gradient")
            grad.append(parse["Gradient"])
        g = np.array(grad)
        if g.shape[0] == 1:
            g = np.squeeze(g)
        return g

    def get_MP2energy(self):
        ens = []
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("MP2 Energy")
            ens.append(parse["MP2 Energy"])
        e = np.array(ens)
        if e.shape[0] == 1:
            e = np.squeeze(e)
        return e

    def get_DipoleMoment(self):
        dips = []
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("Dipole Moment")
            dips.append(parse["Dipole Moment"])
        d = np.array(dips)
        if d.shape[0] == 1:
            d = np.squeeze(d)
        return d

    def get_DipoleDerivatives(self):
        derivs = []
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("Dipole Derivatives")
            derivs.append(parse["Dipole Derivatives"])
        dd = np.array(derivs)
        if dd.shape[0] == 1:
            dd = np.squeeze(dd)
        return dd

    def get_mass(self):
        mass_array = []
        for fchk in self.fchks:
            with GaussianFChkReader(fchk) as reader:
                parse = reader.parse("AtomicMasses")
            mass_array.append(parse["AtomicMasses"])
        ma = np.array(mass_array)
        if ma.shape[0] == 1:
            ma = np.squeeze(ma)
        return ma

