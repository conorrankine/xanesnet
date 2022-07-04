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

###############################################################################
################################## CLASSES ####################################
###############################################################################

class XANES():

    def __init__(
        self,
        e: np.ndarray,
        m: np.ndarray,
        e0: float = None,
        info: dict = None
    ):
        """
        Args:
            e (np.ndarray; 1D): an array of energy (`e`; eV) values
            m (np.ndarray; 1D): an array of intensity (`mu`; arbitrary) values
            e0 (float, optional): the X-ray absorption edge (`e0`; eV) energy.
                If None, an attempt is made to determine `e0` from the maximum
                derivative of `m` with `get_e0()`. Defaults to None.
            info (dict, optional): a dictionary of key/val pairs that can be
                used to store extra information about the XANES spectrum as a
                tag. Defaults to None.

        Raises:
            ValueError: if the `e` and `m` arrays are not the same length.
        """
        
        if len(e) == len(m):
            self._e = e
            self._m = m
        else:
            raise ValueError('the energy (`e`) and XANES spectral intensity '\
                '(`m`) arrays are not the same length')

        if e0 is not None:
            self._e0 = e0
        else:
            self._e0 = self.estimate_e0()

        if info is not None:
            self.info = info
        else:
            self.info = {}

    def estimate_e0(self) -> float:
        """
        Estimates the X-ray absorption edge (`e0`; eV) energy as the energy
        `e` where the derivative of `m` is largest.

        Returns:
            float: the X-ray absorption edge (`e0`; eV) energy.
        """

        return self._e[np.argmax(np.gradient(self._m))]

    @property
    def e(self) -> float:
        return self._e

    @property
    def m(self) -> float:
        return self._m

    @property
    def e0(self) -> float:
        return self._e0

    @property
    def spectrum(self) -> tuple:
        return (self._e, self._m)
