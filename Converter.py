class Constants:
    # conversions from [unit] to a.u. equivalent
    atomic_units = {
        "wavenumbers": 4.55634e-6,
        "angstroms": 1 / 0.529177,
        "amu": 1.000000000000000000 / 6.02213670000e23 / 9.10938970000e-28
    }

    masses = {
        "H": (1.00782503223, "amu"),
        "O": (15.99491561957, "amu"),
        "D": (2.0141017778, "amu")
    }

    @classmethod
    def convert(cls, val, unit, to_AU=True):
        vv = cls.atomic_units[unit]
        return (val * vv) if to_AU else (val / vv)

    @classmethod
    def mass(cls, atom, to_AU=True):
        m = cls.masses[atom]
        if to_AU:
            m = cls.convert(*m)
        return m
