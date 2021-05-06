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

class WACSFDescr():

    def __init__(self, r_max, fc_kind = 'cos', g2_vars = None, g4_vars = None):

        self.r_max = r_max
        self.fc_kind = fc_kind

        self.g2 = True if g2_vars else False
        if self.g2:
            self.g2_eta, self.g2_mu = g2_vars

        self.g4 = True if g4_vars else False
        if self.g4:
            self.g4_eta, self.g4_mu, self.g4_zeta, self.g4_lambda = g4_vars

    def describe(self, system):

        rij_in_range = system.get_distances(0, range(len(system))) < self.r_max
        system = system[rij_in_range]

        ij = [[0,j] for j in range(1, len(system))]
        if len(ij) < 1:
            str_ = ('insufficient atoms within {:.2f} A of the absorption '
                    'site to set up non-zero radial symmetry functions')
            raise RuntimeError(str_.format(self.r_max))
        else:
            ij = np.array(ij, dtype = 'uint16')
        
        if self.g4:
            jik = [[j,0,k] for j in range(1, len(system))
                   for k in range(1, len(system)) if k > j]
            if len(jik) < 1:
                str_ = ('insufficient atoms within {:.2f} A of the absorption '
                        'site to set up non-zero angular symmetry functions')
                raise RuntimeError(str_.format(self.r_max))
            else:
                jik = np.array(jik, dtype = 'uint16')
        
        rij = system.get_distances(ij[:,0], ij[:,1])
        wacsf = self.__get_g1(rij)

        if self.g2:
            zi = system.get_atomic_numbers()[ij[:,0]]
            zj = system.get_atomic_numbers()[ij[:,1]]
            rij = system.get_distances(ij[:,0], ij[:,1])
            g2 = self.__get_g2(zi, zj, rij)
            wacsf = np.append(wacsf, g2)

        if self.g4:
            zi = system.get_atomic_numbers()[jik[:,1]]
            zj = system.get_atomic_numbers()[jik[:,0]]
            zk = system.get_atomic_numbers()[jik[:,2]]
            rij = system.get_distances(jik[:,1], jik[:,0])
            rik = system.get_distances(jik[:,1], jik[:,2])
            rjk = system.get_distances(jik[:,0], jik[:,2])
            ajik = system.get_angles(jik)
            g4 = self.__get_g4(zi, zj, zk, rij, rik, rjk, ajik)
            wacsf = np.append(wacsf, g4)

        return wacsf

    def __fc(self, rij):

        if self.fc_kind == 'tanh':
            return np.power(np.tanh(1.0 - (rij / self.r_max)), 3.0)
        else:
            return (np.cos((np.pi * rij) / self.r_max) + 1.0) / 2.0

    def __r(self, rij, a, b):

        return np.exp(-1.0 * a * np.square(rij[:,np.newaxis] - b))

    def __a(self, ajik, a, b):

        return np.power(1.0 + (a * np.cos(np.radians(ajik))[:,np.newaxis]), b)

    def __get_g1(self, rij):

        g1 = np.sum(self.__fc(rij))

        return g1

    def __get_g2(self, zi, zj, rij):

        g2 = (np.sum((zj * self.__fc(rij))[:,np.newaxis] 
                     * self.__r(rij, self.g2_eta, self.g2_mu), axis = 0))

        return g2

    def __get_g4(self, zi, zj, zk, rij, rik, rjk, ajik):

        g4 = (np.sum((zj * zk * self.__fc(rij) * self.__fc(rik)
                     * self.__fc(rjk))[:,np.newaxis]
                     * self.__r(rij, self.g4_eta, self.g4_mu)
                     * self.__r(rik, self.g4_eta, self.g4_mu)
                     * self.__r(rjk, self.g4_eta, self.g4_mu)
                     * self.__a(ajik, self.g4_lambda, self.g4_zeta), axis = 0)
        ) * np.power(2.0, 1.0 - self.g4_zeta)

        return g4

    @property
    def r_max(self):
        
        return self.__r_max 

    @property
    def fc_kind(self):
        
        return self.__fc_kind

    @property
    def g2_eta(self):
        
        return self.__g2_eta

    @property
    def g2_mu(self):
        
        return self.__g2_mu

    @property
    def g4_eta(self):
        
        return self.__g4_eta

    @property
    def g4_mu(self):
        
        return self.__g4_mu

    @property
    def g4_zeta(self):
        
        return self.__g4_zeta

    @property
    def g4_lambda(self):
        
        return self.__g4_lambda

    @r_max.setter
    def r_max(self, r_max):

        if not isinstance(r_max, float) or r_max <= 0:
            str_ = 'r_max has to be a) a float, and b) > 0; got {}'
            raise ValueError(str_.format(r_max))
        else:
            self.__r_max = r_max

    @fc_kind.setter
    def fc_kind(self, fc_kind):

        if not isinstance(fc_kind, str) or fc_kind not in ['cos', 'tanh']:
            str_ = 'fc_kind has to be \'cos\' or \'tanh\'; got {}'
            raise ValueError(str_.format(fc_kind))
        else:
            self.__fc_kind = fc_kind

    @g2_eta.setter
    def g2_eta(self, g2_eta):

        if not isinstance(g2_eta, (np.ndarray, list)):
            str_ = 'g2_eta has to be a) a Numpy array or b) a list; got {}'
            raise ValueError(str_.format(g2_eta))
        else:
            self.__g2_eta = (g2_eta if isinstance(g2_eta, np.ndarray)
                             else np.array(g2_eta, dtype = 'float32'))

    @g2_mu.setter
    def g2_mu(self, g2_mu):

        if not isinstance(g2_mu, (np.ndarray, list)):
            str_ = 'g2_mu has to be a) a Numpy array or b) a list; got {}'
            raise ValueError(str_.format(g2_mu))
        else:
            self.__g2_mu = (g2_mu if isinstance(g2_mu, np.ndarray)
                            else np.array(g2_mu, dtype = 'float32'))

    @g4_eta.setter
    def g4_eta(self, g4_eta):

        if not isinstance(g4_eta, (np.ndarray, list)):
            str_ = 'g4_eta has to be a) a Numpy array or b) a list; got {}'
            raise ValueError(str_.format(g4_eta))
        else:
            self.__g4_eta = (g4_eta if isinstance(g4_eta, np.ndarray)
                             else np.array(g4_eta, dtype = 'float32'))

    @g4_mu.setter
    def g4_mu(self, g4_mu):

        if not isinstance(g4_mu, (np.ndarray, list)):
            str_ = 'g4_mu has to be a) a Numpy array or b) a list; got {}'
            raise ValueError(str_.format(g4_mu))
        else:
            self.__g4_mu = (g4_mu if isinstance(g4_mu, np.ndarray)
                            else np.array(g4_mu, dtype = 'float32'))

    @g4_zeta.setter
    def g4_zeta(self, g4_zeta):

        if not isinstance(g4_zeta, (np.ndarray, list)):
            str_ = 'g4_zeta has to be a) a Numpy array or b) a list; got {}'
            raise ValueError(str_.format(g4_zeta))
        else:
            self.__g4_zeta = (g4_zeta if isinstance(g4_zeta, np.ndarray)
                              else np.array(g4_zeta, dtype = 'float32'))

    @g4_lambda.setter
    def g4_lambda(self, g4_lambda):

        if not isinstance(g4_lambda, (np.ndarray, list)):
            str_ = 'g4_lambda has to be a) a Numpy array or b) a list; got {}'
            raise ValueError(str_.format(g4_lambda))
        else:
            self.__g4_lambda = (g4_lambda if isinstance(g4_lambda, np.ndarray)
                                else np.array(g4_lambda, dtype = 'float32'))