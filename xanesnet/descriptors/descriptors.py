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

from .generic import _Descriptor
from .rdc import RDC
from .wacsf import WACSF

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def get_descriptor(
    descriptor_type: str,
    params: dict = None
) -> _Descriptor:
    """
    Returns a descriptor (`_Descriptor`) instance of the specified type,
    optionally initialised with a set of parameters that can be passed through
    to the constructor function of the descriptor to override the defaults.

    Args:
        descriptor_type (str): Descriptor type, e.g., 'rdc' (radial
            distribution functions); 'wacsf' (weighted atom-centred symmetry
            functions); etc.
        params (dict, optional): Parameters passed through to the constructor
            function of the descriptor. Defaults to None.

    Raises:
        ValueError: If `descriptor_type` is not an available/valid descriptor.

    Returns:
        _Descriptor: Descriptor.
    """
    
    if params is None:
        params = {}

    descriptors = {
        'rdc': RDC,
        'wacsf': WACSF
    }

    try:
        return descriptors.get(descriptor_type)(**params)
    except KeyError:
        raise ValueError(
            f'\'{descriptor_type}\' is not an available/valid descriptor'
        ) from None
