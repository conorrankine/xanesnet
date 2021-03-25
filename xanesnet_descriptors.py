###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np
import tensorflow as tf
import itertools as itr

from ase import Atoms

###############################################################################
################################ DESCRIPTORS ##################################
###############################################################################

class CoulombMatrixDescr():

    def __init__(self, n_max, sort = True, triu = True):

        self.n_max = n_max
        self.sort = sort
        self.triu = triu

    def describe(self, system):

        system = system[:self.n_max]

        z = system.get_atomic_numbers()
        p = system.get_positions()
        
        zij = z[:,np.newaxis] * z
        rij = np.sqrt(np.sum(np.square(p[:,np.newaxis] - p), axis = -1))
        with np.errstate(divide = 'ignore'):
            cmat = zij / rij
        np.fill_diagonal(cmat, 0.5 * np.power(z, 2.4))

        if self.n_max > len(cmat):
            cmat = self.__pad_cmat(cmat)

        if self.sort:
            cmat = self.__sort_cmat(cmat)

        if self.triu:
            cmat = self.__triu_cmat(cmat)

        return cmat

    def __sort_cmat(self, cmat):

        v = np.linalg.norm(cmat, axis = 0)
        v_sort_idx = np.flip(np.argsort(v))
        cmat_sort = cmat[v_sort_idx][:,v_sort_idx]

        return cmat_sort

    def __triu_cmat(self, cmat):

        cmat_triu = cmat[np.triu_indices(self.n_max)]

        return cmat_triu

    def __pad_cmat(self, cmat):

        v = self.n_max - len(cmat)
        cmat_pad = np.pad(cmat, ((0, v), (0, v)), 'constant')

        return cmat_pad
                    
    @property
    def n_max(self):
        
        return self.__n_max

    @n_max.setter
    def n_max(self, n_max):

        if not isinstance(n_max, int) or n_max <= 1:
            str_ = 'n_max has to be a) an integer, and b) > 1; got {}'
            raise ValueError(str_.format(n_max))
        else:
            self.__n_max = n_max

class RadialDistCurveDescr():

    def __init__(self, r_max, dr, alpha):

        self.r_max = r_max
        self.d_max = 2.0 * self.r_max
        self.dr = dr
        self.alpha = alpha
        self.r = np.linspace(0.0, self.d_max, int(self.d_max / self.dr), 
                             dtype = 'float64')

    def describe(self, system):

        rij_in_range = system.get_distances(0, range(len(system))) < self.r_max

        if len(rij_in_range) <= 1:
            str_ = 'no atoms within {:.0f} Ang. of the absorption site'
            raise RuntimeError(str_.format(self.r_max))
        else:
            system = system[rij_in_range]

        z = system.get_atomic_numbers()
        
        at_pairs = np.array(list(itr.combinations(range(z.size), 2)), 
                            dtype = 'uint16')

        zi = z[at_pairs[:,0]]
        zj = z[at_pairs[:,1]]
        zij = (zi * zj)      
        rij = system.get_distances(at_pairs[:,0], at_pairs[:,1])
        rij_r_sq = np.square(rij[:, np.newaxis] - self.r)
        exp = np.exp(-1.0 * self.alpha * rij_r_sq)
        rdc = np.sum(zij[:, np.newaxis] * exp, axis = 0)

        return rdc

    @property
    def r_max(self):
        
        return self.__r_max

    @property
    def dr(self):
        
        return self.__dr

    @property
    def alpha(self):
        
        return self.__alpha

    @r_max.setter
    def r_max(self, r_max):

        if not isinstance(r_max, float) or r_max <= 0:
            str_ = 'r_max has to be a) a float, and b) > 0; got {}'
            raise ValueError(str_.format(r_max))
        else:
            self.__r_max = r_max

    @dr.setter
    def dr(self, dr):

        if not isinstance(dr, float) or dr > self.r_max:
            str_ = 'dr has to be a) a float, and b) > r_max; got {}'
            raise ValueError(str_.format(dr))
        else:
            self.__dr = dr

    @alpha.setter
    def alpha(self, alpha):

        if not isinstance(alpha, float) or alpha <= 0.0:
            str_ = 'alpha has to be a) a float, and b) > 0; got {}'
            raise ValueError(str_.format(alpha))
        else:
            self.__alpha = alpha