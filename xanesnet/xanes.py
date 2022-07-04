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
                derivative of `m` with `get_e0()`.
                Defaults to None.
            info (dict, optional): a dictionary of key/val pairs that can be
                used to store extra information about the XANES spectrum as a
                tag.
                Defaults to None.

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

    def scale(
        self,
        fit_limits: tuple = (100.0, 400.0),
        flatten: bool = True
    ):
        """
        Scales the intensity (`mu`; arbitrary) values using the 'edge-step'
        approach: fitting a 2nd-order (quadratic) polynomial, `fit`, to the
        post-edge (where `e` >= `e0`; eV), determining the 'edge-step',
        `fit(e0)`, and scaling `mu` by dividing through by this value. `mu`
        can also be flattened; in this case, the post-edge is levelled off
        to ca. 1.0 by adding (1.0 - `fit(e0)`) to `mu` where `e` >= `e0`.

        Args:
            fit_limits (tuple, optional): lower and upper limits (in eV
                relative to the X-ray absorption edge; `e0`) defining the `e`
                window over which the 2nd-order (quadratic) polynomial, `fit`,
                is determined.
                Defaults to (100.0, 400.0).
            flatten (bool, optional): toggles flattening of the post-edge by
                adding (1.0 - `fit(e0)`) to `mu` where `e` >= `e0`.
                Defaults to True.
        """

        e_rel = self._e - self._e0
        e_rel_min, e_rel_max = fit_limits

        fit_window = (e_rel >= e_rel_min) & (e_rel <= e_rel_max)

        fit = np.polynomial.Polynomial.fit(
            self._e[fit_window],
            self._m[fit_window],
            deg = 2
        )

        self._m /= fit(self._e0)

        if flatten:
            self._m[self._e >= self._e0] += (
                1.0 - (fit(self._e)[self._e >= self._e0] / fit(self._e0))
            )

        return self   

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
