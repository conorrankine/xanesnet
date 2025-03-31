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
from typing import Any
from abc import ABC, abstractmethod

###############################################################################
################################## CLASSES ####################################
###############################################################################

class BaseTransformer(ABC):
    """
    An abstract base class for transformers used in, e.g., `dataset.py` to
    transform input objects into (1D) np.ndarrays.

    This class defines a template for compatible transformers; subclasses are
    expected to implement the `.transform()` method to carry out the
    object -> (1D) np.ndarray transformation (as the exact transformation will
    be object- and implementation-specific) and the `.size` property to provide
    advance access to the number of elements in the (1D) np.ndarray
    representation returned by the `.transform()` method.
    """

    def __init__(self):
        
        pass

    @abstractmethod
    def transform(
        self,
        obj: Any
    ) -> np.ndarray:
        """
        Transforms an input object into a (1D) np.ndarray representation; the
        exact transformation will be object- and implementation-specific. 

        Args:
            obj (Any): Object to transform.

        Returns:
            np.ndarray: (1D) np.ndarray representation.
        """
        
        pass 

    @property
    @abstractmethod
    def size(
        self
    ) -> int:
        """
        Returns:
            int: Number of elements in the (1D) np.ndarray representation
                returned by the `.transform()` method.
        """

        pass
    