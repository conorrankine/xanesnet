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
from argparse import ArgumentParser

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################

def parse_args(args: list):

    p = ArgumentParser()

    p.add_argument('-v', '--version', action = 'version', 
        version = xanesnet.__version__)
    
    sub_p = p.add_subparsers(dest = 'mode')

    learn_p = sub_p.add_parser('learn')
    learn_p.add_argument('inp_f', type = str, 
        help = 'path to .json input file w/ variable definitions')
    learn_p.add_argument('--no-save', dest = 'save', action = 'store_false',
        help = 'toggles model directory creation and population to <off>')

    predict_p = sub_p.add_parser('predict')
    predict_p.add_argument('mdl_dir', type = str, 
        help = 'path to populated model directory')
    predict_p.add_argument('xyz_dir', type = str, 
        help = 'path to .xyz input directory for prediction')
    
    args = p.parse_args()

    return args  

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

def main(args: list):

    pass

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main()

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################