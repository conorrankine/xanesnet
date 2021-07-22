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
from abc import ABC
from abc import abstractmethod
from abc import abstractproperty

###############################################################################
################################## CLASSES ####################################
###############################################################################

class VectorDescriptor(ABC):
    """
    An abstract base class for transforming a molecular system into a 
    fingerprint feature vector, or 'descriptor', that encodes the local 
    environment around an absorption site as a vector.
    """
    
    def __init__(
        self, 
        r_min: float = 0.0, 
        r_max: float = 6.0
    ):
        """
        Args:
            r_min (float): The minimum radial cutoff distance (in A) around
                the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float): The maximum radial cutoff distance (in A) around 
                the absorption site.
                Defaults to 6.0.
        """

        if isinstance(r_min, (int, float)) and r_min >= 0.0:
            self.r_min = float(r_min)
        else:
            raise ValueError(f'expected r_min: int/float >== 0.0; got {r_min}')

        if isinstance(r_max, (int, float)) and r_max > r_min:
            self.r_max = float(r_max)
        else:
            raise ValueError(f'expected r_max: int/float > r_min; got {r_max}')        

    @abstractmethod
    def transform(
        self, 
        system: Atoms
    ) -> np.ndarray:
        """
        Transforms a molecular system into a fingerprint feature vector, or
        'descriptor', that encodes the local environment around an absorption 
        site as a vector; the absorption site has to be the first atom defined 
        for the molecular system.

        Args:
            system (Atoms): A molecular system.

        Returns:
            np.ndarray: A fingerprint feature vector for the molecular system.
        """
        
        pass 

class RadDistCurve(VectorDescriptor):
    """
    A class for transforming a molecular system into a radial (or 'pair') 
    distribution curve (RDCs). The RDC is - simplistically - like a histogram
    of pairwise internuclear distances discretised over an auxilliary 
    real-space grid and smoothed out using Gaussians; pairs are made between 
    the absorption site and all atoms within a defined radial cutoff.
    """

    def __init__(
        self, 
        r_min: float = 0.0,
        r_max: float = 6.0,
        dr: float = 0.01,
        alpha: float = 10.0
    ):
        """
        Args:
            r_min (float): The minimum radial cutoff distance (in A) around
                the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float): The maximum radial cutoff distance (in A) around 
                the absorption site.
                Defaults to 6.0.
            dr (float): The step size (in A) for the auxilliary real-space grid
                that the RDC is discretised over.
                Defaults to 0.01.
            alpha (float): A smoothing parameter used in a Gaussian exponent 
                that defines the effective spatial resolution of the RDC.
                Defaults to 10.0.
        """

        super().__init__(r_min, r_max)
        
        if isinstance(dr, (int, float)) and r_max >= dr > 0.0:
            self.dr = float(dr)
        else:
            raise ValueError(f'expected dr: int/float > 0.0; got {dr}')

        if isinstance(alpha, (int, float)) and alpha > 0.0:
            self.alpha = float(alpha)
        else:
            raise ValueError(f'expected alpha: int/float > 0.0; got {alpha}')

        nr_aux = int(np.absolute(self.r_max - self.r_min) / self.dr) + 1
        self.r_aux = np.linspace(self.r_min, self.r_max, nr_aux)

    def transform(
        self, 
        system: Atoms
    ) -> np.ndarray:
        """
        Transforms a molecular system into an RDC descriptor that encodes
        the local environment around an absorption site; the absorption site 
        has to be the first atom defined for the molecular system.

        Args:
            system (Atoms): A molecular system.

        Returns:
            np.ndarray: An RDC descriptor for the molecular system.
        """
        
        if not isinstance(system, Atoms):
            raise TypeError(f'systems passed as arguments to .transform ',
                'should be ase.Atoms objects')

        rij_in_range = system.get_distances(0, range(len(system))) < self.r_max
        system = system[rij_in_range]
        
        ij = [[0,j] for j in range(1, len(system))]
        if len(ij) < 1:
            raise RuntimeError(f'too few atoms within {self.r_max:.2f} A of ',
                'the absorption site to set up non-zero radial distribution ',
                'curve (no pairs)')
        else:
            ij = np.array(ij, dtype = 'uint16')

        zi = system.get_atomic_numbers()[ij[:,0]]
        zj = system.get_atomic_numbers()[ij[:,1]]
        rij = system.get_distances(ij[:,0], ij[:,1])
        rij_r_sq = np.square(rij[:, np.newaxis] - self.r_aux)
        exp = np.exp(-1.0 * self.alpha * rij_r_sq)
        rdc = np.sum((zi * zj)[:, np.newaxis] * exp, axis = 0)

        return rdc

class WACSF(VectorDescriptor):
    """
    A class for transforming a molecular system into a weighted atom-centered 
    symmetry function (WACSF) descriptor. WACSFs encode the local geometry 
    around an absorption site using parameterised radial and angular 
    components. For reference, check out the following publication:
    
    > J. Chem. Phys., 2018, 148, 241709 (10.1063/1.5019667)
    
    ...which builds on the earlier ACSF descriptor introduced in:
    
    > J. Chem. Phys., 2011, 134, 074106 (10.1063/1.3553717)
    """

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 6.0,
        g2_vars = None,
        g4_vars = None,
        with_weighting = True
    ):
        """
        Args:
            r_min (float): The minimum radial cutoff distance (in A) around
                the absorption site; should be 0.0.
                Defaults to 0.0.
            r_max (float): The maximum radial cutoff distance (in A) around 
                the absorption site.
                Defaults to 6.0.
            g2_vars (list, optional): A list of eta (h) and mu (m) pairs,
                i.e. [[h1,m1],[h2,m2],...,[hn,mn]], used to construct G2
                (radial) symmetry functions.
                Defaults to None.
            g4_vars (list, optional): A list of eta (h), mu (m), lambda (l),
                and zeta (z) quintuples, i.e. [[h1,m1,l1,z1],[h2,m2,l2,z2],
                ...,[hn,mn,ln,zn]], used to construct G4 (angular) symmetry
                functions.
                Defaults to None.
        """

        super().__init__(r_min, r_max)

        self.use_g2 = bool(g2_vars)
        if self.use_g2:
            g2_vars_ = np.array(g2_vars, dtype = 'float32')
            try:
                self.g2_h, self.g2_m = g2_vars_.T
            except ValueError:
                raise ValueError(f'g2_vars is not formatted properly; ',
                    'expected [[h1,m1],[h2,m2],...,[hn,mn]]')

        self.use_g4 = bool(g4_vars)
        if self.use_g4:
            g4_vars_ = np.array(g4_vars, dtype = 'float32')
            try:
                self.g4_h, self.g4_m, self.g4_l, self.g4_z = g4_vars_.T
            except ValueError:
                raise ValueError(f'g4_vars is not formatted properly; ',
                    'expected [[h1,m1,l1,z1],[h2,m2,l2,z2],...,[hn,mn,ln,zn]]')

        if isinstance(with_weighting, bool):
            self.with_weighting = with_weighting
        else:
            raise ValueError(f'expected with_weighting: bool (True/False); ',
                'got {with_weighting}')

    def transform(
        self, 
        system: Atoms
    ) -> np.ndarray:
        """
        Transforms a molecular system into a WACSF descriptor that encodes the
        local environment around an absorption site; the absorption site has
        to be the first atom defined for the molecular system.

        Args:
            system (Atoms): A molecular system.

        Returns:
            np.ndarray: A WACSF descriptor for the molecular system.
        """

        if not isinstance(system, Atoms):
            raise TypeError(f'systems passed as arguments to .transform ',
                'should be ase.Atoms objects')

        rij_in_range = system.get_distances(0, range(len(system))) < self.r_max
        system = system[rij_in_range]

        ij = [[0,j] for j in range(1, len(system))]
        if len(ij) < 1:
            raise RuntimeError(f'too few atoms within {self.r_max:.2f} A of ',
                'the absorption site to set up non-zero radial symmetry ',
                'functions (no pairs)')
        else:
            ij = np.array(ij, dtype = 'uint16')
        
        if self.use_g4:
            jik = [[j,0,k] for j in range(1, len(system))
                   for k in range(1, len(system)) if k > j]
            if len(jik) < 1:
                raise RuntimeError(f'too few atoms within {self.r_max:.2f} A ',
                    'of the absorption site to set up non-zero angular ',
                    'symmetry functions (no triples)')
            else:
                jik = np.array(jik, dtype = 'uint16')
        
        rij = system.get_distances(ij[:,0], ij[:,1])
        wacsf = self._g1(rij)

        if self.use_g2:
            if self.with_weighting:
                zi = system.get_atomic_numbers()[ij[:,0]] 
                zj = system.get_atomic_numbers()[ij[:,1]]
            else:
                zi = zj = 1.0
            rij = system.get_distances(ij[:,0], ij[:,1])
            g2 = self._g2(zi, zj, rij)
            wacsf = np.append(wacsf, g2)

        if self.use_g4:
            if self.with_weighting:
                zi = system.get_atomic_numbers()[jik[:,1]]
                zj = system.get_atomic_numbers()[jik[:,0]]
                zk = system.get_atomic_numbers()[jik[:,2]]
            else:
                zi = zj = zk = 1.0
            rij = system.get_distances(jik[:,1], jik[:,0])
            rik = system.get_distances(jik[:,1], jik[:,2])
            rjk = system.get_distances(jik[:,0], jik[:,2])
            ajik = system.get_angles(jik)
            g4 = self._g4(zi, zj, zk, rij, rik, rjk, ajik)
            wacsf = np.append(wacsf, g4)

        return wacsf

    def _fc(self, rij):

        return (np.cos((np.pi * rij) / self.r_max) + 1.0) / 2.0

    def _r(self, rij, a, b):

        return np.exp(-1.0 * a * np.square(rij[:,np.newaxis] - b))

    def _a(self, ajik, a, b):

        return np.power(1.0 + (a * np.cos(np.radians(ajik))[:,np.newaxis]), b)

    def _g1(self, rij):

        g1 = np.sum(self._fc(rij))

        return g1

    def _g2(self, zi, zj, rij):

        g2 = (np.sum((zj * self._fc(rij))[:,np.newaxis] 
            * self._r(rij, self.g2_h, self.g2_m), axis = 0))

        return g2

    def _g4(self, zi, zj, zk, rij, rik, rjk, ajik):

        g4 = (np.sum((zj * zk * self._fc(rij) * self._fc(rik)
            * self._fc(rjk))[:,np.newaxis]
            * self._r(rij, self.g4_h, self.g4_m)
            * self._r(rik, self.g4_h, self.g4_m)
            * self._r(rjk, self.g4_h, self.g4_m)
            * self._a(ajik, self.g4_l, self.g4_z), axis = 0)
        ) * np.power(2.0, 1.0 - self.g4_z)

        return g4