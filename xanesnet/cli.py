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

import xanesnet
from argparse import ArgumentParser, Namespace
from pathlib import Path

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################

def parse_args() -> Namespace:

    p = ArgumentParser()

    p.add_argument(
        '--version', '-v', action = 'version', version = xanesnet.__version__
    )
    
    sub_p = p.add_subparsers(
        dest = 'mode'
    )

    train_p = sub_p.add_parser(
        'train',
        help = 'train a model'                       
    )
    train_p.add_argument(
        'data_src', type = Path, nargs = '+', 
        help = 'path to the input (X) and output (Y) data source(s)'
    )
    train_p.add_argument(
        '--config', '-c', type = Path, default = None,
        help = 'path to a .yaml configurational file'
    )

    predict_p = sub_p.add_parser(
        'predict',
        help = 'make predictions using your trained model'
    )
    predict_p.add_argument(
        'data_src', type = Path,
        help = 'path to the input (X) data source'
    )
    predict_p.add_argument(
        'model', type = Path,
        help = 'path to the trained model'
    )

    args = p.parse_args()

    return args  

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

def main():

    args = parse_args()

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main()

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################