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

import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace
from ase.io import write
from xanesnet.fdmnes import read_fdmnes_out

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################

def parse_args() -> Namespace:
    """
    Parses command line arguments for `bav_to_xyz.py`.

    Returns:
        argparse.Namespace: Parsed command line arguments as an
        argparse.Namespace object that holds the arguments as attributes.
    """

    p = ArgumentParser()

    p.add_argument('bav_f', type = Path,
        help = '`_bav.txt` file from a complete FDMNES calculation'
    )
    p.add_argument('--output_f', '-o', type = Path, default = None,
        help = 'output (.xyz) file for writing Cartesian coordinates to'
    )

    args = p.parse_args()

    return args

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

def main() -> None:

    args = parse_args()

    with open(args.bav_f, 'r') as f:
        atoms = read_fdmnes_out(f)

    if args.output_f is None:
        write(sys.stdout, atoms, format = 'xyz', fmt = '%12.8f')
    else:
        with open(args.output_f, 'w') as f:
            write(f, atoms, format = 'xyz', fmt = '%12.8f')

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main()

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################
