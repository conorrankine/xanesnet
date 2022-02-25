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

from abc import ABC
from abc import abstractmethod
from abc import abstractproperty

###############################################################################
################################## CLASSES ####################################
###############################################################################

class XANESScaler(ABC):
    """
    An abstract base class for scaling/normalising XANES spectra.
    """

    def __init__(
        self
    ):
        
        pass

    @abstractmethod
    def transform(self, m: np.ndarray) -> np.ndarray:
        """
        Scales a XANES spectrum.

        Args:
            m (np.ndarray): A XANES spectrum.

        Returns:
            np.ndarray: A scaled/normalised XANES spectrum.
        """

        pass

class SimpleXANESScaler(XANESScaler):
    """
    A class for scaling XANES spectra in the simplest way; dividing the
    XANES spectral intensities by the intensity of the final point in the
    XANES spectrum. This is a very blunt approach to normalisation!
    """

    def __init__(
        self
    ):

        super().__init__()

    def transform(self, m: np.ndarray) -> np.ndarray:

        return m / m[-1]

class EdgeStepXANESScaler(XANESScaler):
    """
    A class for scaling XANES spectra formally using the 'edge-step' approach;
    fitting a 2nd-order (quadratic) polynomial, q, to the post-edge part of the
    XANES spectrum, determining the 'edge-step', q(e_edge), and scaling the
    XANES spectrum by dividing through by this value. The XANES spectra can
    also be flattened; in this case, the post-edge part of the XANES spectrum
    is levelled off to 1.0 by adding 1.0 - q(e_edge) where e > e_edge.
    """

    def __init__(
        self
    ):
        
        super().__init__()

    def transform(self, m: np.ndarray) -> np.ndarray:
        #TODO: implement this!
        
        err_str = 'work in progress; coming to XANESNET soon!'
        raise NotImplementedError(err_str)