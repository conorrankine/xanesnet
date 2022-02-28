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
    An abstract base class for scaling XANES spectra.
    """

    def __init__(
        self
    ):
        
        pass

    @abstractmethod
    def transform(self, e: np.ndarray, m: np.ndarray,) -> np.ndarray:
        """
        Scales a XANES spectrum.

        Args:
            e (np.ndarray): Energy values (in eV).
            m (np.ndarray): Intensity values (in arb. units).
            
        Returns:
            np.ndarray: Intensity values (in arb. units) after scaling.
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

    def transform(self, e: np.ndarray, m: np.ndarray) -> np.ndarray:

        m_scaled = m / m[-1]

        return m_scaled

class EdgeStepXANESScaler(XANESScaler):
    """
    A class for scaling XANES spectra formally using the 'edge-step' approach;
    fitting a 2nd-order (quadratic) polynomial, q, to the post-edge part of the
    XANES spectrum, determining the 'edge-step', q(e_edge), and scaling the
    XANES spectrum by dividing through by this value. The XANES spectra can
    also be flattened; in this case, the post-edge part of the XANES spectrum
    is levelled off to 1.0 by adding (1.0 - q(e_edge)) where e > e_edge.
    """

    def __init__(
        self,
        e_edge: float,
        e_fit_min_rel: float = 100.0,
        e_fit_max_rel: float = 400.0,
        flatten = True,
    ):
        """
        Args:
            e_edge (float): The absorption edge energy (in eV); available @
                <http://skuld.bmsc.washington.edu/scatter/AS_periodic.html>
            e_fit_min_rel (float, optional): The minimum energy (in eV,
                relative to the absorption edge) for the quadratic 
                polynomial fitting window.
                Defaults to 100.0.
            e_fit_max_rel (float, optional): The maximum energy (in eV,
                relative to the absorption edge) for the quadratic
                polynomial fitting window.
                Defaults to 400.0.
            flatten (bool, optional): Toggles flattening of the post-edge
                part of the XANES spectrum; if True, it is levelled off to 1.0
                by adding (1.0 - q(e_edge)) where e > e_edge.
                Defaults to True.
        """
        
        super().__init__()

        self.e_edge = e_edge
        self.e_fit_min = e_edge + e_fit_min_rel
        self.e_fit_max = e_edge + e_fit_max_rel
        self.flatten = flatten

    def transform(self, e: np.ndarray, m: np.ndarray) -> np.ndarray:

        e_edge_idx = np.argmin(np.abs(e - self.e_edge))
        e_fit_min_idx = np.argmin(np.abs(e - self.e_fit_min))
        e_fit_max_idx = np.argmin(np.abs(e - self.e_fit_max))

        q = np.polynomial.Polynomial.fit(
            e[e_fit_min_idx:e_fit_max_idx],
            m[e_fit_min_idx:e_fit_max_idx],
            deg = 2
        )

        edge_step = q(self.e_edge)

        m_scaled = m / edge_step

        if self.flatten:
            m_scaled[e_edge_idx:] += (1.0 - (q(e) / edge_step)[e_edge_idx:])

        return m_scaled
