"""
XANESNET-REDUX
Copyright (C) 2025  Conor D. Rankine

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
from abc import ABC, abstractmethod
from ..templates import BaseTransformer

###############################################################################
################################## CLASSES ####################################
###############################################################################

class Descriptor(BaseTransformer, ABC):
    """
    An abstract base class for a descriptor, i.e. a featuriser that transforms
    a set of atoms into a set of features.
    """

    def __init__():

            pass
    
    @abstractmethod
    def transform(
        self,
        atoms: Atoms
    ) -> np.ndarray:
        """
        Transforms a set of atoms into a set of features that encode the local
        environment around an atom of interest; the atom of interest is assumed
        to be the first atom in the set of atoms.

        Args:
            atoms (Atoms): Set of atoms.

        Returns:
            np.ndarray: Set of features.
        """
        
        pass 

    @property
    @abstractmethod
    def size(
        self
    ) -> int:
        """
        Returns:
            int: Number of features.
        """

        pass

class VectorDescriptor(Descriptor, ABC):
    """
    An abstract base class for a vector descriptor, i.e. a featuriser that
    transforms a set of atoms into a one-dimensional vector of features.
    """
    
    def __init__(
        self, 
        r_min: float, 
        r_max: float
    ):
        """
        Args:
            r_min (float): The minimum radial distance around the X-ray
                absorption site (in Angstroem) to use for featurisation.
            r_max (float): The maximum radial distance around the X-ray
                absorption site (in Angstroem) to use for featurisation.
        """

        if r_min >= 0.0:
            self.r_min = float(r_min)
        else:
            raise ValueError(
                f'expected r_min: float >== 0.0; got {r_min}'
            )

        if r_max > r_min:
            self.r_max = float(r_max)
        else:
            raise ValueError(
                f'expected r_max: float > r_min; got {r_max}'
            )        
