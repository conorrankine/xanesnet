"""
XANESNET
Copyright (C) 2021  Conor D. Rankine

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software 
Foundation, either Version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np
from ase import Atoms

###############################################################################
################################ DESCRIPTORS ##################################
###############################################################################

class CoulombMatrix():
    """
    A class for transforming molecular systems into Coulomb matrices; Coulomb
    matrices are described in the following publication:
    
    > Phys. Rev. Lett., 2012, 108, 058301 (10.1103/PhysRevLett.108.058301)
    """

    def __init__(self, n_max: int, sort = True, triu = True, **kwargs):
        """
        Args:
            n_max (int): The maximum number of atoms around (and inclusive of)
                         the absorption site to take into account in the 
                         Coulomb matrix (which will have dimensions 
                         n_max * n_max unless flattened).
            sort (bool, optional): Toggles sorting the rows/columns of the 
                                   Coulomb matrix according to the L2 norms. 
                                   Defaults to True.
            triu (bool, optional): Toggles flattening the Coulomb matrix to the
                                   elements of the upper triangle. 
                                   Defaults to True.
        """

        self.n_max = n_max
        self.sort = sort
        self.triu = triu

    def describe(self, system: Atoms) -> np.ndarray:
        """
        Transforms a molecular system into a Coulomb matrix descriptor using a
        fast vectorised approach.

        Args:
            system (ase.Atoms): A molecular system. 

        Returns:
            np.ndarray: A Coulomb matrix descriptor for the molecular system.
        """

        if not isinstance(system, Atoms):
            raise TypeError(f'systems passed as arguments to .describe should ',
                             'be ase.Atoms objects')

        system = system[:self.n_max]

        z = system.get_atomic_numbers()
        p = system.get_positions()
        
        zij = z[:,np.newaxis] * z
        rij = np.sqrt(np.sum(np.square(p[:,np.newaxis] - p), axis = -1))
        with np.errstate(divide = 'ignore'):
            cmat = zij / rij
        np.fill_diagonal(cmat, 0.5 * np.power(z, 2.4))

        if self.n_max > len(cmat):
            cmat = self._pad_cmat(cmat)

        if self.sort:
            cmat = self._sort_cmat(cmat)

        if self.triu:
            cmat = self._triu_cmat(cmat)

        return cmat

    def _sort_cmat(self, cmat):

        v = np.linalg.norm(cmat, axis = 0)
        v_sort_idx = np.flip(np.argsort(v))
        cmat_sort = cmat[v_sort_idx][:,v_sort_idx]

        return cmat_sort

    def _triu_cmat(self, cmat):

        cmat_triu = cmat[np.triu_indices(self.n_max)]

        return cmat_triu

    def _pad_cmat(self, cmat):

        v = self.n_max - len(cmat)
        cmat_pad = np.pad(cmat, ((0, v), (0, v)), 'constant')

        return cmat_pad
                    
    @property
    def n_max(self):
        
        return self.__n_max

    @n_max.setter
    def n_max(self, n_max):

        if not isinstance(n_max, int):
            raise TypeError(f'n_max has to be an integer; got {n_max}')
        elif not n_max > 1:
            raise ValueError(f'n_max has to be > 1; got {n_max}')
        else:
            self.__n_max = n_max

class RadDistCurve():
    """
    A class for transforming molecular systems into radial (or 'pair') 
    distribution curves (RDCs). The RDC is - simplistically - like a histogram
    of pairwise internuclear distances discretised over a real-space grid and
    phenonmenologically broadened using Gaussians; pairs are made between the
    absorption site and all atoms within a defined radial cutoff.
    """

    def __init__(self, r_max: float, gridsize: int, alpha: float, **kwargs):
        """
        Args:
            r_max (float): The maximum radial cutoff distance (in Angstrom)
                           around the absorption site; the RDC is defined 
                           from zero to r_max.
            gridsize (int): The number of gridpoints over which the RDC (which
                            will have a length = gridsize) is discretised.
            alpha (float): A phenomenological broadening parameter that defines
                           the effective spatial resolution of the RDC.
        """

        self.r_max = r_max
        self.gridsize = gridsize
        self.alpha = alpha
        self.r = np.linspace(0.0, self.r_max, self.gridsize, dtype = 'float32')

    def describe(self, system: Atoms) -> np.ndarray:
        """
        Transforms a molecular system into an RDC descriptor using a fast
        vectorised approach. The absorption site has to be the first atom
        defined for the molecular system.

        Args:
            system (Atoms): A molecular system.

        Returns:
            np.ndarray: An RDC descriptor for the molecular system.
        """
        
        if not isinstance(system, Atoms):
            raise TypeError(f'systems passed as arguments to .describe should ',
                             'be ase.Atoms objects')

        rij_in_range = system.get_distances(0, range(len(system))) < self.r_max
        system = system[rij_in_range]
        
        ij = [[0,j] for j in range(1, len(system))]
        if len(ij) < 1:
            errstr = (f'too few atoms within {self.r_max:.2f} A of the ',
                       'absorption site to set up a non-zero radial ',
                       'distribution curve')
            raise RuntimeError(errstr)
        else:
            ij = np.array(ij, dtype = 'uint16')

        zi = system.get_atomic_numbers()[ij[:,0]]
        zj = system.get_atomic_numbers()[ij[:,1]]
        rij = system.get_distances(ij[:,0], ij[:,1])
        rij_r_sq = np.square(rij[:, np.newaxis] - self.r)
        exp = np.exp(-1.0 * self.alpha * rij_r_sq)
        rdc = np.sum((zi * zj)[:, np.newaxis] * exp, axis = 0)

        return rdc

    @property
    def r_max(self):
        
        return self.__r_max

    @r_max.setter
    def r_max(self, r_max):

        if not isinstance(r_max, (int, float)):
            raise TypeError(f'r_max has to be an integer/float; got {r_max}')
        elif not r_max > 0:
            raise ValueError(f'r_max has to be > 0; got {r_max}')
        else:
            self.__r_max = float(r_max)

    @property
    def gridsize(self):
        
        return self.__gridsize

    @gridsize.setter
    def gridsize(self, gridsize):

        if not isinstance(gridsize, int):
            raise TypeError(f'gridsize has to be an integer; got {gridsize}')
        elif not gridsize > 1:
            raise ValueError(f'gridsize has to be > 1; got {gridsize}')
        else:
            self.__gridsize = gridsize

    @property
    def alpha(self):
        
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):

        if not isinstance(alpha, (int, float)):
            raise TypeError(f'alpha has to be an integer/float; got {alpha}')
        elif not alpha > 0:
            raise ValueError(f'alpha has to be > 0; got {alpha}')
        else:
            self.__alpha = float(alpha)

class WACSF():
    """
    A class for transforming molecular systems into weighted atom-centered
    symmetry function (WACSF) descriptors. WACSF descriptors encode the 
    local environment around an absorption site using parameterised 
    radial and angular components. For reference, check out the following 
    publication:
    
    > J. Chem. Phys., 2018, 148, 241709 (10.1063/1.5019667)
    
    ...which builds on the earlier ACSF descriptor introduced in:
    
    > J. Chem. Phys., 2011, 134, 074106 (10.1063/1.3553717)
    """

    def __init__(self, r_max: float, g2_vars = None, g4_vars = None, **kwargs):
        """
        Args:
            r_max (float): The maximum radial cutoff distance (in Angstrom)
                           around the absorption site.
            g2_vars (list, optional): A list of two elements: i) a list
                                      /np.ndarray of 'eta' values, and ii) a
                                      list/np.ndarray of 'mu' values. These are
                                      used to construct the 'G2' radial WACSF
                                      components. len('eta') == len('mu'). 
                                      Defaults to None.
            g4_vars (list, optional): A list of four elements: i) a list
                                      /np.ndarray of 'eta' values, ii) a
                                      list/np.ndarray of 'mu' values, iii) a 
                                      list/np.ndarray of 'zeta' values, and iv)
                                      a list/np.ndarray of 'lambda' values.                                     
                                      These are used to construct the 'G4' 
                                      angular WACSF components. len('eta') == 
                                      len('mu') == len('zeta') == len('lambda). 
                                      Defaults to None.
        """

        self.r_max = r_max

        self.g2 = True if g2_vars else False
        if self.g2:
            self.g2_eta, self.g2_mu = g2_vars

        self.g4 = True if g4_vars else False
        if self.g4:
            self.g4_eta, self.g4_mu, self.g4_zeta, self.g4_lambda = g4_vars

    def describe(self, system: Atoms) -> np.ndarray:
        """
        Transforms a molecular system into a WACSF descriptor using a fast
        vectorised approach. The absorption site has to be the first atom
        defined for the molecular system.

        Args:
            system (Atoms): A molecular system.

        Returns:
            np.ndarray: A WACSF descriptor for the molecular system.
        """

        rij_in_range = system.get_distances(0, range(len(system))) < self.r_max
        system = system[rij_in_range]

        ij = [[0,j] for j in range(1, len(system))]
        if len(ij) < 1:
            errstr = (f'too few atoms within {self.r_max:.2f} A of the ',
                       'absorption site to set up non-zero radial ',
                       'symmetry functions')
            raise RuntimeError(errstr)
        else:
            ij = np.array(ij, dtype = 'uint16')
        
        if self.g4:
            jik = [[j,0,k] for j in range(1, len(system))
                   for k in range(1, len(system)) if k > j]
            if len(jik) < 1:
                errstr = (f'too few atoms within {self.r_max:.2f} A of ',
                           'the absorption site to set up non-zero angular ',
                           'symmetry functions')
                raise RuntimeError(errstr)
            else:
                jik = np.array(jik, dtype = 'uint16')
        
        rij = system.get_distances(ij[:,0], ij[:,1])
        wacsf = self._get_g1(rij)

        if self.g2:
            zi = system.get_atomic_numbers()[ij[:,0]]
            zj = system.get_atomic_numbers()[ij[:,1]]
            rij = system.get_distances(ij[:,0], ij[:,1])
            g2 = self._get_g2(zi, zj, rij)
            wacsf = np.append(wacsf, g2)

        if self.g4:
            zi = system.get_atomic_numbers()[jik[:,1]]
            zj = system.get_atomic_numbers()[jik[:,0]]
            zk = system.get_atomic_numbers()[jik[:,2]]
            rij = system.get_distances(jik[:,1], jik[:,0])
            rik = system.get_distances(jik[:,1], jik[:,2])
            rjk = system.get_distances(jik[:,0], jik[:,2])
            ajik = system.get_angles(jik)
            g4 = self._get_g4(zi, zj, zk, rij, rik, rjk, ajik)
            wacsf = np.append(wacsf, g4)

        return wacsf

    def _fc(self, rij):

        return (np.cos((np.pi * rij) / self.r_max) + 1.0) / 2.0

    def _r(self, rij, a, b):

        return np.exp(-1.0 * a * np.square(rij[:,np.newaxis] - b))

    def _a(self, ajik, a, b):

        return np.power(1.0 + (a * np.cos(np.radians(ajik))[:,np.newaxis]), b)

    def _get_g1(self, rij):

        g1 = np.sum(self._fc(rij))

        return g1

    def _get_g2(self, zi, zj, rij):

        g2 = (np.sum((zj * self._fc(rij))[:,np.newaxis] 
                     * self._r(rij, self.g2_eta, self.g2_mu), axis = 0))

        return g2

    def _get_g4(self, zi, zj, zk, rij, rik, rjk, ajik):

        g4 = (np.sum((zj * zk * self._fc(rij) * self._fc(rik)
                     * self._fc(rjk))[:,np.newaxis]
                     * self._r(rij, self.g4_eta, self.g4_mu)
                     * self._r(rik, self.g4_eta, self.g4_mu)
                     * self._r(rjk, self.g4_eta, self.g4_mu)
                     * self._a(ajik, self.g4_lambda, self.g4_zeta), axis = 0)
        ) * np.power(2.0, 1.0 - self.g4_zeta)

        return g4

    @property
    def r_max(self):
        
        return self.__r_max 

    @r_max.setter
    def r_max(self, r_max):

        if not isinstance(r_max, (int, float)):
            raise TypeError(f'r_max has to be an integer/float; got {r_max}')
        elif not r_max > 0:
            raise ValueError(f'r_max has to be > 0; got {r_max}')
        else:
            self.__r_max = float(r_max)
            
    @property
    def g2_eta(self):
        
        return self.__g2_eta

    @g2_eta.setter
    def g2_eta(self, g2_eta):

        if not isinstance(g2_eta, (np.ndarray, list)):
            raise TypeError(f'g2_eta has to be an np.ndarray/list; ',
                             'got {g2_eta}')
        else:
            self.__g2_eta = (g2_eta if isinstance(g2_eta, np.ndarray)
                             else np.array(g2_eta, dtype = 'float32'))

    @property
    def g2_mu(self):
        
        return self.__g2_mu

    @g2_mu.setter
    def g2_mu(self, g2_mu):

        if not isinstance(g2_mu, (np.ndarray, list)):
            raise TypeError(f'g2_mu has to be an np.ndarray/list; ',
                             'got {g2_mu}')
        else:
            self.__g2_mu = (g2_mu if isinstance(g2_mu, np.ndarray)
                            else np.array(g2_mu, dtype = 'float32'))

    @property
    def g4_eta(self):
        
        return self.__g4_eta

    @g4_eta.setter
    def g4_eta(self, g4_eta):

        if not isinstance(g4_eta, (np.ndarray, list)):
            raise TypeError(f'g4_eta has to be an np.ndarray/list; ',
                             'got {g4_eta}')
        else:
            self.__g4_eta = (g4_eta if isinstance(g4_eta, np.ndarray)
                             else np.array(g4_eta, dtype = 'float32'))
    @property
    def g4_mu(self):
        
        return self.__g4_mu

    @g4_mu.setter
    def g4_mu(self, g4_mu):

        if not isinstance(g4_mu, (np.ndarray, list)):
            raise TypeError(f'g4_mu has to be an np.ndarray/list; ',
                             'got {g4_mu}')
        else:
            self.__g4_mu = (g4_mu if isinstance(g4_mu, np.ndarray)
                            else np.array(g4_mu, dtype = 'float32'))
            
    @property
    def g4_zeta(self):
        
        return self.__g4_zeta

    @g4_zeta.setter
    def g4_zeta(self, g4_zeta):

        if not isinstance(g4_zeta, (np.ndarray, list)):
            raise TypeError(f'g4_zeta has to be an np.ndarray/list; ',
                             'got {g4_zeta}')
        else:
            self.__g4_zeta = (g4_zeta if isinstance(g4_zeta, np.ndarray)
                              else np.array(g4_zeta, dtype = 'float32'))
            
    @property
    def g4_lambda(self):
        
        return self.__g4_lambda

    @g4_lambda.setter
    def g4_lambda(self, g4_lambda):

        if not isinstance(g4_lambda, (np.ndarray, list)):
            raise TypeError(f'g4_lambda has to be an np.ndarray/list; ',
                             'got {g4_lambda}')
        else:
            self.__g4_lambda = (g4_lambda if isinstance(g4_lambda, np.ndarray)
                                else np.array(g4_lambda, dtype = 'float32'))