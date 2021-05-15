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
################################## CONVOLUTE ##################################
###############################################################################

class ArctanConvoluter():
    """
    A class for convoluting (broadening) theoretical + predicted XANES spectra
    to account for phenomenological effects like core-hole lifetime broadening,
    instrumental resolution, and many-body effects (e.g. inelastic losses). 
    This is achieved approximately using an empirical model similar to the 
    Seah-Dench formalism, originally described in the following publication:
    
    > NPL Rep. Chem., 1978, 82. 
    
    The implementation here is inspired by the energy-dependent arctan
    convolution model implemented in the FDMNES program package 
    (<http://fdmnes.neel.cnrs.fr/>) and the equations are outlined in the 
    following publications:
    
    > J. Phys. Chem. A, 2020, 124, 4263-4270 (10.1021/acs.jpca.0c03723)
    
    > Molecules, 2020, 25, 2715 (10.3390/molecules25112715)
    """

    def __init__(self, e: np.ndarray, e_edge: float, e_l: float, e_c: float, 
                 e_f: float, g_hole: float, g_max: float, p_dim = 150):
        """
        Args:
            e (np.ndarray): The energy grid over which the arctan convolution
                            function is defined and operates. The energy grid 
                            is expected to have equally-spaced gridpoints 
                            (e.g. a np.linspace)
            e_edge (float): The absorption edge energy (in eV).
            e_l (float): The width of the arctan convolution function (in eV).
            e_c (float): The center of the arctan convolution function (in eV).
            e_f (float): The Fermi energy (in eV).
            g_hole (float): The core state width (in eV).
            g_max (float): The final state width (in eV).
            p_dim (int, optional): The number of additional energy grid points
                                   to pad both ends of the energy grid with 
                                   before arctan convolution. Defaults to 150.
        """
 
        self.e = e
        self.e_edge = e_edge
        self.e_l = e_l
        self.e_c = e_c
        self.e_f = e_f
        self.g_hole = g_hole
        self.g_max = g_max
        self.p_dim = p_dim

        e_pad_min = np.min(self.e) - (np.diff(self.e)[0] * self.p_dim)
        e_pad_max = np.max(self.e) + (np.diff(self.e)[0] * self.p_dim)
        self.e_pad = np.pad(self.e, self.p_dim, mode = 'linear_ramp',
                        end_values = (e_pad_min, e_pad_max))

        self.g = self._arctan_g()
        self.h = self._lorentz_h()

    def convolute(self, mu: np.ndarray) -> np.ndarray:
        """
        Convolutes a XANES spectrum with an energy-dependent arctan function
        to account for phenomenological effects like core-hole lifetime 
        broadening, instrumental resolution, and many-body effects (e.g. 
        inelastic losses).

        Args:
            mu (np.ndarray): An unconvoluted XANES spectrum. The XANES spectrum
                             is expected to be defined over the energy grid
                             (ArctanConvoluter.e), i.e. every gridpoint in the
                             energy grid pairs up with every point in mu, and
                             len(ArctanConvoluter.e) == len(mu).

        Returns:
            np.ndarray: An arctan-convoluted XANES spectrum. The XANES spectrum
                        is defined over the energy grid (ArctanConvoluter.e).
        """

        if not isinstance(mu, np.ndarray):
            raise TypeError(f'mu has to be a np.ndarray; got {mu}')
        elif not len(mu) == len(self.e):
            raise ValueError((f'dimension of mu ({len(mu)}) does not match ',
                              f'dimension of the energy grid ({len(self.e)})'))

        # pad mu at both ends of the energy grid
        mu_pad = np.pad(mu, self.p_dim, mode = 'edge')
        # squash mu below the absorption edge to eliminate pre-edge peaks
        mu_pad_squash = np.where(self.e_pad < self.e_edge, 0.0, mu_pad)
        # convolute mu with the energy-dependent arctan convolution function
        mu_conv = np.sum(self.h.T * mu_pad_squash, axis = 1)
        # scale mu
        mu_conv = (np.sum(mu_pad_squash) * mu_conv) / np.sum(mu_conv)
        # unpad mu at both ends of the energy grid
        mu_conv = mu_conv[self.p_dim:-self.p_dim]

        return mu_conv
                      
    def _arctan_g(self):

        e_rel = self.e_pad - self.e_edge
        gq = (e_rel - self.e_f) / self.e_c
        
        g = self.g_hole + (
            self.g_max * (
                0.5 + (
                    (1.0 / np.pi) * (
                        np.arctan((np.pi / 3.0) * (self.g_max / self.e_l) 
                                  * (gq - (1.0 / np.square(gq)))
                        )
                    )
                )
            )
        )

        return g

    def _lorentz_h(self):

        de = self.e_pad[:,np.newaxis] - self.e_pad[np.newaxis,:]

        h = (0.5 * self.g) / (np.square(de) + np.square(0.5 * self.g))
        h = np.where(de != 0.0, h, 1.0)

        return h

    @property
    def e(self):
        
        return self.__e

    @e.setter
    def e(self, e):

        if not isinstance(e, np.ndarray):
            raise TypeError(f'e has to be a np.ndarray; got {e}')
        else:
            self.__e = e

    @property
    def e_edge(self):
        
        return self.__e_edge

    @e_edge.setter
    def e_edge(self, e_edge):

        if not isinstance(e_edge, (int, float)):
            raise TypeError(f'e_edge has to be an integer/float; got {e_edge}')
        elif not e_edge > 0:
            raise ValueError(f'e_edge has to be > 0; got {e_edge}')
        else:
            self.__e_edge = float(e_edge)

    @property
    def e_l(self):
        
        return self.__e_l

    @e_l.setter
    def e_l(self, e_l):

        if not isinstance(e_l, (int, float)):
            raise TypeError(f'e_l has to be an integer/float; got {e_l}')
        elif not e_l > 0:
            raise ValueError(f'e_l has to be > 0; got {e_l}')
        else:
            self.__e_l = float(e_l)

    @property
    def e_c(self):
        
        return self.__e_c

    @e_c.setter
    def e_c(self, e_c):

        if not isinstance(e_c, (int, float)):
            raise TypeError(f'e_c has to be an integer/float; got {e_c}')
        elif not e_c > 0:
            raise ValueError(f'e_c has to be > 0; got {e_c}')
        else:
            self.__e_c = float(e_c)

    @property
    def e_f(self):
        
        return self.__e_f

    @e_f.setter
    def e_f(self, e_f):

        if not isinstance(e_f, (int, float)):
            raise TypeError(f'e_f has to be an integer/float; got {e_f}')
        else:
            self.__e_f = float(e_f)

    @property
    def g_hole(self):
        
        return self.__g_hole

    @g_hole.setter
    def g_hole(self, g_hole):

        if not isinstance(g_hole, (int, float)):
            raise TypeError(f'g_hole has to be an integer/float; got {g_hole}')
        else:
            self.__g_hole = float(g_hole)

    @property
    def g_max(self):
        
        return self.__g_max

    @g_max.setter
    def g_max(self, g_max):

        if not isinstance(g_max, (int, float)):
            raise ValueError(f'g_max has to be an integer/float; got {g_max}')
        else:
            self.__g_max = float(g_max)

    @property
    def p_dim(self):
        
        return self.__p_dim

    @p_dim.setter
    def p_dim(self, p_dim):

        if not isinstance(p_dim, int):
            raise ValueError(f'p_dim has to be an integer; got {p_dim}')
        else:
            self.__p_dim = p_dim