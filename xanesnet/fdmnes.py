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
from ase import Atom, Atoms
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

def read_fdmnes_in(f: TextIO) -> Atoms:
    """
    Reads an input file for the FDMNES program and returns an atomic
    configuration (ASE Atoms object).

    Args:
        f (TextIO): FDMNES input file.

    Returns:
        Atoms: Atomic configuration (ASE Atoms object).
    """

    atoms = Atoms()
    
    for line in _readlines_between(f, 'MOLECULE', 'END'):
        line_ = line.strip().split()
        if len(line_) == 4:
            atom = Atom(int(line_[0]), [float(x) for x in line_[1:]])
            atoms += atom

    return atoms

def read_fdmnes_out(f: TextIO) -> Atoms:
    """
    Reads an output ('_bav') file from the FDMNES program and returns an atomic
    configuration (ASE Atoms object).

    Args:
        f (TextIO): FDMNES output ('_bav') file.

    Returns:
        Atoms: Atomic configuration (ASE Atoms object).
    """

    atoms = Atoms()

    for line in _readlines_between(f, 'POSITIONS', 'IAPOT'):
        line_ = line.strip().split()
        if len(line_) == 11:
            atom = Atom(int(line_[0]), [float(x) for x in line_[1:4]])
            atoms += atom

    return atoms

def _readlines_between(f: TextIO, start_str: str, end_str: str) -> str:
    """
    Yields lines from a file found between lines containing a first
    (`start_str`) and second (`end_str`) marker string exclusively (i.e. the
    lines containing `start_str` and `end_str` are not included); the search
    for the marker strings is case-insensitive.

    Args:
        f (TextIO): File.
        start_str (str): String marking the start of the section to read.
        end_str (str): String marking the end of the section to read.

    Yields:
        str: Lines found between `start_str` and `end_str`.
    """

    read = False
    for line in f.readlines():
        if end_str.lower() in line.lower():
            read = False
        if read:
            yield line
        if start_str.lower() in line.lower():
            read = True
