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
################################### CLASSES ###################################
###############################################################################

class Convoluter(ABC):
    """
    An abstract base class for convoluting XANES spectra to account for
    phenomenological effects like core-hole lifetime broadening, instrumental
    resolution, and many-body effects (e.g. inelastic losses); convolution is
    carried out with a Lorentzian kernel over an auxilliary energy grid
    (e_min -> e_max : de) defined relative to an absorption edge (e_edge).
    """

    def __init__(
        self,
        e_edge: float,
        e_fermi: float = -5.0, 
        e_min: float = -50.0,
        e_max: float = 150.0,
        de: float = 0.2
    ):
        """
        Args:
            e_edge (float): The absorption edge energy (in eV); available @
                <http://skuld.bmsc.washington.edu/scatter/AS_periodic.html>
            e_fermi (float): The Fermi energy (in eV, relative to the
                absorption edge); cross-sectional contributions from the
                occupied states below the Fermi energy are removed.
                Defaults to -5.0 eV.
            e_min (float): The minimum energy (in eV, relative to the
                absorption edge) for the auxilliary energy grid that the
                convoluter operates over.
                Defaults to -50.0 eV.
            e_max (float): The maximum energy (in eV, relative to the
                absorption edge) for the auxilliary energy grid that the
                convoluter operates over.
                Defaults to +150.0 eV.
            de (float): The step size (in eV) for the auxilliary energy grid
                that the convoluter operates over.
                Defaults to 0.2 eV.
        """
 
        self.e_edge = float(e_edge)
        self.e_fermi = float(e_fermi)
        self.e_min = float(e_min)
        self.e_max = float(e_max)
        self.de = float(de)
        
        ne_aux = int(np.absolute(self.e_max - self.e_min) / self.de) + 1
        self.e_aux = np.linspace(self.e_min, self.e_max, ne_aux)

        lorentz = lambda x, x0, g: g * (0.5 / ((x - x0)**2 + (0.5 * g)**2))

        self.conv_kernel = lorentz(*np.meshgrid(self.e_aux, self.e_aux),
            self._get_lorentz_width()
        )

    def convolute(self, e: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        This function convolutes XANES spectra to account for phenomenological
        effects like core-hole lifetime broadening, instrumental resolution,
        and many-body effects (e.g. inelastic losses); convolution is
        carried out with a Lorentzian kernel over an auxilliary energy grid
        (e_min -> e_max : de) defined relative to an absorption edge (e_edge).

        Args:
            e (np.ndarray): Energy values (in eV).
            m (np.ndarray): Intensity values (in arb. units).

        Returns:
            (np.ndarray): Intensity values (in arb. units) after convolution.
        """

        # copy energy (e) and intensity values (m) for temporary transformation
        e_ = e.copy()
        m_ = m.copy()

        # rescale energy values (e_) rel. to the absorption edge (self.e_edge)
        e_ -= self.e_edge
        
        # if the rescaled energy values (e_) do not map onto the auxilliary
        # energy grid that the arctangent convoluter operates over, the
        # intensity values (m_) will have to be projected
        try:
            projection = True if not np.allclose(e_, self.e_aux) else False
        except ValueError:
            projection = True

        if projection:
            # project intensity values (m_) onto auxilliary energy grid
            m_ = np.interp(self.e_aux, e_, m_)

        # flatten intensity values (m_) below the Fermi energy; remove the
        # cross-sectional contributions from the occupied states
        m_ = np.where(self.e_aux < self.e_fermi, 0.0, m_)

        # apply convolution kernel
        m_ = np.sum(self.conv_kernel * m_, axis = 1)
        
        if projection:
            # project intensity values (m_) off auxilliary energy grid
            m_ = np.interp(e_, self.e_aux, m_)

        return m_
    
    @abstractmethod
    def _get_lorentz_width(self) -> np.ndarray:
        # returns the width(s) for the Lorentzian kernel; can return either a
        # i) singular value for convolution with a Lorentzian kernel of fixed
        # width, or ii) np.ndarray for convolution with a Lorentzian kernel of
        # energy-dependent width over the auxilliary energy grid (self.e)
   
        pass

class FixedGammaConvoluter(Convoluter):
    """
    A class for convoluting XANES spectra with a fixed-width convolution model
    to account for phenomenological effects like core-hole lifetime broadening,
    instrumental resolution, and many-body effects (e.g. inelastic losses);
    convolution is carried out with a Lorentzian kernel over an auxilliary
    energy grid (e_min -> e_max : de) defined relative to an absorption edge
    (e_edge).
    
    The width of the Lorentzian kernel (g_hole) is fixed over the auxilliary
    energy grid. The implementation here is similiar to the implementation in
    the FDMNES program package (<http://fdmnes.neel.cnrs.fr/>); see p.43-48 in
    the FDMNES user manual (Section C: Convolution) for additional details.
    """

    def __init__(
        self,
        e_edge: float,
        e_fermi: float = -5.0,
        e_min: float = -50.0,
        e_max: float = 150.0,
        de: float = 0.2,
        g_hole: float = 2.0
    ):
        """
        Args:
            e_edge (float): The absorption edge energy (in eV); available @
                <http://skuld.bmsc.washington.edu/scatter/AS_periodic.html>
            e_fermi (float): The Fermi energy (in eV, relative to the
                absorption edge); cross-sectional contributions from the
                occupied states below the Fermi energy are removed.
                Defaults to -5.0 eV.
            e_min (float): The minimum energy (in eV, relative to the
                absorption edge) for the auxilliary energy grid that the
                fixed gamma convoluter operates over.
                Defaults to -50.0 eV.
            e_max (float): The maximum energy (in eV, relative to the
                absorption edge) for the auxilliary energy grid that the
                fixed gamma convoluter operates over.
                Defaults to +150.0 eV.
            de (float): The step size (in eV) for the auxilliary energy grid
                that the fixed gamma convoluter operates over.
                Defaults to 0.2 eV.
            g_hole (float): The fixed gamma convolutional width (in eV).
                Defaults to 2.0 eV.
        """

        self.g_hole = float(g_hole)

        super().__init__(
            float(e_edge),
            float(e_fermi), 
            float(e_min), 
            float(e_max), 
            float(de)
        )

    def _get_lorentz_width(self) -> float:
        # returns fixed gamma convolutional width; see p.43-48 in the
        # FDMNES user manual (Section C: Convolution) for additional details

        return self.g_hole

class ArctanConvoluter(Convoluter):
    """
    A class for convoluting XANES spectra with an energy-dependent arctangent
    convolution model to account for phenomenological effects like core-hole
    lifetime broadening, instrumental resolution, and many-body effects (e.g.
    inelastic losses); convolution is carried out with a Lorentzian kernel
    over an auxilliary energy grid (e_min -> e_max : de) defined relative to
    an absorption edge (e_edge).
    
    The width of the Lorentzian kernel at each point on the auxilliary energy
    grid is determined with a parameterised energy-dependent arctangent
    function. The implementation here is similiar to the implementation in the
    FDMNES program package (<http://fdmnes.neel.cnrs.fr/>); see p.43-48 in the
    FDMNES user manual (Section C: Convolution) for additional details.
    """
    
    def __init__(
        self, 
        e_edge: float,
        e_fermi: float = -5.0,
        e_min: float = -50.0,
        e_max: float = 150.0,
        de: float = 0.2,
        e_l: float = 30.0, 
        e_c: float = 30.0,
        g_hole: float = 2.0,
        g_m: float = 15.0
    ):
        """
        Args:
            e_edge (float): The absorption edge energy (in eV); available @
                <http://skuld.bmsc.washington.edu/scatter/AS_periodic.html>
            e_fermi (float): The Fermi energy (in eV, relative to the
                absorption edge); cross-sectional contributions from the
                occupied states below the Fermi energy are removed.
                Defaults to -5.0 eV.
            e_min (float): The minimum energy (in eV, relative to the
                absorption edge) for the auxilliary energy grid that the
                arctangent convoluter operates over.
                Defaults to -50.0 eV.
            e_max (float): The maximum energy (in eV, relative to the
                absorption edge) for the auxilliary energy grid that the
                arctangent convoluter operates over.
                Defaults to +150.0 eV.
            de (float): The step size (in eV) for the auxilliary energy grid
                that the arctangent convoluter operates over.
                Defaults to 0.2 eV.
            e_l (float): The width of the arctan convolution function (in eV).
                Defaults to 30.0 eV.
            e_c (float): The center of the arctan convolution function (in eV).
                Defaults to 30.0 eV.
            g_hole (float): The core state width (in eV).
                Defaults to 2.0 eV.
            g_m (float): The final state width (in eV).
                Defaults to 15.0 eV.
        """
 
        self.e_l = float(e_l)
        self.e_c = float(e_c)
        self.g_hole = float(g_hole)
        self.g_m = float(g_m)
        
        super().__init__(
            float(e_edge),
            float(e_fermi), 
            float(e_min), 
            float(e_max), 
            float(de)
        )
        
    def _get_lorentz_width(self) -> np.ndarray:
        # returns an energy-dependent arctangent function; see p.43-48 in the
        # FDMNES user manual (Section C: Convolution) for additional details
   
        e = (self.e_aux - self.e_fermi) / self.e_c
        
        with np.errstate(divide = 'ignore'):
            arctan = (np.pi / 3.0) * (self.g_m / self.e_l) * (e - (1.0 / e**2))

        g = self.g_hole + self.g_m * (
            (1.0 / 2.0) + (1.0 / np.pi) * np.arctan(arctan)
        )
        
        return g