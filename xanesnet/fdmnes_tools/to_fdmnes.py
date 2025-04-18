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

from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Union
from ase.io import read
from xanesnet.fdmnes import write_fdmnes_in

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################

def parse_args() -> Namespace:
    """
    Parses command line arguments for `to_fdmnes.py`.

    Returns:
        argparse.Namespace: Parsed command line arguments as an
        argparse.Namespace object that holds the arguments as attributes.
    """

    p = ArgumentParser()

    p.add_argument('input_f', type = Path,
        help = 'input file containing an atomic configuration (e.g., .xyz)'
    )
    p.add_argument('--absorber', '-a', type = _absorber, default = 1,
        help =('index for the absorbing atom')
    )
    p.add_argument('--edge', '-e', type = str, default = 'K',
        help = ('label for the absorption edge (e.g., K, L, M, etc.)')
    )
    p.add_argument('--radius', '-r', type = float, default = 6.0,
        help = ('radius of the atomic cluster in Angstroem')
    )
    p.add_argument('--range', '-q', type = str, default = 'auto',
        help = ('energy range for the calculation in eV; the expected format '
            'is \'START STEP STOP\' (e.g. \'-30.0 0.1, 60.0\') although if '
            '\'auto\' is given the energy range is determined appropriately '
            'for the absorption edge')
    )

    args = p.parse_args()

    return args

def _absorber(a: str) -> Union[int, str]:
    """
    Custom argument type for the `--absorber`/`-a` command line argument to
    allow the user to specify the absorber as i) an integer corresponding to
    the atomic index, or ii) a string corresponding to the atomic symbol.

    Args:
        a (str): Input value of the `--absorber`/`-a` argument.

    Returns:
        Union[int, str]: Integer, if int(`a`) passes, else String if int(`a`)
            returns ValueError.
    """

    try:
        return int(a)
    except ValueError:
        return a

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

def main() -> None:

    args = parse_args()

    with open(args.input_f, 'r') as f:
        atoms = read(f)

    if isinstance(args.absorber, str):
        args.absorber = atoms.get_chemical_symbols().index(args.absorber) + 1

    if args.range == 'auto':
        auto_ranges = {
            'K': '-30.0 0.2 60.0 0.5 120.0 1.0 300.0',
            'L23': '-15.0 0.1 30.0 0.2 45.0 0.5 60.0'
        }
        try:
            args.range = auto_ranges[args.edge]
        except KeyError:
            raise NotImplementedError(
                'automatic energy range selection isn\'t available at the '
                '{} edge; set the energy range explicity.'.format(args.edge)
            ) from None
        
    params = {
        'absorber': args.absorber,
        'edge': args.edge,
        'radius': args.radius,
        'range': args.range,
        'energpho': None,
        'scf': None,
        'green': None,
        'quadrupole': None,
        'convolution': None
    }

    if args.edge == 'L23':
        params.update({'spinorbit': None})

    with open(f'{args.input_f.stem}.in', 'w') as f:
        write_fdmnes_in(f, atoms, **params)

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main()

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################
