"""
XANESNET
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

from pathlib import Path
from ase import Atoms
from typing import TextIO

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def write_fdmnes_in(f: TextIO, atoms: Atoms, **params) -> None:
    """
    Writes an input file for the FDMNES program to carry out an XAS or XES
    calculation on a specified atomic configuration (ASE Atoms object).

    Args:
        f (TextIO): FDMNES input file.
        atoms (Atoms): Atomic configuration (ASE Atoms object).
        **params (dict): A dictionary of keyword/value pairs specifying
            additional input to be added to the FDMNES input file; if the
            value associated with a keyword is None then the keyword is added
            to the FDMNES input file without a value, otherwise the value is
            written on the line immediately following the keyword.
    """
    

    f.write(f'FILEOUT\n./{Path(f.name).stem}\n\n')

    for key, val in params.items():
        if val is not None:
            f.write(f'{key.upper()}\n{val}\n')
        else:
            f.write(f'{key.upper()}\n')
    f.write('\n')

    f.write('MOLECULE\n1.0 1.0 1.0 90.0 90.0 90.0\n')
    for atom in atoms:
        f.write('{:<6.0f}\t{:>12.8f}\t{:>12.8f}\t{:>12.8f}\n'.format(
            atom.number, *atom.position)
        )
    f.write('END')
