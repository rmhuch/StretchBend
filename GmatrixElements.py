import numpy as np

class GmatrixStretchBend:
    @classmethod
    def calc_Gphiphi(cls, m1, m2, m3, r12, r23, phi123):
        term1 = 1 / (m1 * r12**2)
        term2 = 1 / (m3 * r23**2)
        term3 = (1 / m2) * ((1 / r12**2) + (1 / r23**2) - (2 * np.cos(phi123) / (r12 * r23)))
        return term1 + term2 + term3

    @classmethod
    def calc_Grr(cls, m1, m2):
        """g_rr for r12, r12"""
        term1 = (1 / m1) + (1 / m2)
        return term1

    @classmethod
    def calc_Grrprime(cls, m1, phi123):
        """g_rr for r12, r13"""
        term1 = (1 / m1)
        term2 = np.cos(phi123)
        return term1 * term2

    @classmethod
    def calc_Grphi(cls, m2, r23, phi123):
        """g_rr for r12, phi123"""
        term1 = 1 / (m2 * r23)
        term2 = np.sin(phi123)
        return -term1 * term2
