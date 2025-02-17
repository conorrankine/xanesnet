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
from argparse import ArgumentParser, Namespace
from typing import TextIO, Generator
from ase.io import iread, write

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################

def parse_args() -> Namespace:
    """
    Parses command line arguments for `pull_complexes.py`.

    Returns:
        argparse.Namespace: Parsed command line arguments as an
        argparse.Namespace object that holds the arguments as attributes.
    """

    p = ArgumentParser()

    p.add_argument('tmqm_X_f', type = Path,
        help = '`tmQM_X.xyz` file from the tmQM distribution'
    )
    p.add_argument('element', type = str,
        help =('pull complexes only if they contain the specified element')
    )

    args = p.parse_args()

    return args

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def _get_comments(f: TextIO) -> Generator[str, None, None]:

    for line in f:
        if len(line.split()) > 4:
            yield line.strip()

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

def main() -> None:

    args = parse_args()

    with open(args.tmqm_X_f, 'r') as f_in:    
        for comment, atoms in zip(_get_comments(f_in), iread(f_in)):
            csd_code = comment.split()[2]
            if args.element.capitalize() in atoms.get_chemical_symbols():
                with open(f'{csd_code}.xyz', 'w') as f_out:
                    write(f_out, atoms, comment = comment)

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main()

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################
