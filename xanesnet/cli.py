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

import sys as sys
import ast as ast

from argparse import ArgumentParser

import xanesnet

from xanesnet.core import learn
from xanesnet.core import predict

###############################################################################
############################ CLI ARGUMENT PARSING #############################
###############################################################################

def parse_args(args: list):
    # loads a list of command-line arguments into a namespace via argparser

    p = ArgumentParser()
    
    sub_p = p.add_subparsers(dest = 'mode')

    learn_p = sub_p.add_parser('learn')
    learn_p.add_argument('inp_f', type = str, 
        help = ('path to a .txt input file with variable definitions ',
                '(see examples & docs)'))

    predict_p = sub_p.add_parser('predict')
    predict_p.add_argument('mdl_dir', type = str, 
        help = ('path to a model.[?] directory created using learn mode'))
    predict_p.add_argument('xyz_dir', type = str, 
        help = ('path to a directory containing .xyzs to predict with'))
    predict_p.add_argument('-c', '--conv_inp_f', type = str, 
        help = ('path to .txt input file with variable definitions ',
                'for arctan convolution (see examples & docs'))
    
    args = p.parse_args()

    return args  

def load_input_f(inp_f: str) -> dict:
    # loads a plaintext (.txt) input file into a dictionary - key/value pairs
    # are expected to be '='-delimited (e.g. key = value) and entered 
    # one-per-line; type is automatically determined (or otherwise assumed to 
    # be string) and comment lines (starting with '#') are passed over

    print(f'>> loading user input @ {inp_f}\n')
   
    inp = {}
    
    with open(inp_f) as f:
        ls = [l for l in f if l.strip() and not l.startswith('#')]

    for l in ls:
        (var, val) = l.split('=')
        print(f'>> {var.strip()} :: {val.strip()}')
        try:
            inp[var.strip()] = ast.literal_eval(val.strip())
        except ValueError:
            inp[var.strip()] = val.strip()

    print()

    return inp

###############################################################################
################################ MAIN ROUTINE #################################
###############################################################################

def main(args: list):

    if len(args) == 0:
        sys.exit()
    else:
        args = parse_args(args)
        
    ver = xanesnet.__version__

    print(f'\n***************************************************************',
          f'\n***************************************************************',
          f'\n***************************************************************',
          f'\n********                                               ********',
          f'\n********              | X A N E S N E T |              ********',
          f'\n********                                               ********',
          f'\n********                    v {ver}                    ********',
          f'\n********                                               ********',
          f'\n********      Software Design + Development Lead:      ********',
          f'\n********             Dr. Conor D. Rankine              ********',
          f'\n********                                               ********',
          f'\n********                 Science Lead                  ********',
          f'\n********              Dr. Tom J. Penfold               ********',
          f'\n********                                               ********',
          f'\n********        check out (+ cite!) this code:         ********',
          f'\n********                                               ********',
          f'\n********     1) J. Phys. Chem. A, 2020, 124, 4263      ********',
          f'\n********        DOI : 10.1021/acs.jpca.0c03723         ********',
          f'\n********                                               ********',
          f'\n********         2) Molecules, 2020, 25, 2715          ********',
          f'\n********        DOI : 10.3390/molecules25112715        ********',
          f'\n********                                               ********',
          f'\n********   3) Phys. Chem. Chem. Phys, 2021, 23, 9259   ********',
          f'\n********           DOI : 10.1039/D0CP06244H            ********',
          f'\n********                                               ********',
          f'\n***************************************************************',
          f'\n***************************************************************',
          f'\n***************************************************************\n')

    if args.mode == 'learn':
        learn(**load_input_f(args.inp_f))

    if args.mode == 'predict':
        conv_vars = load_input_f(args.conv_inp_f) if args.conv_inp_f else {}
        predict(args.mdl_dir, args.xyz_dir, conv_vars = conv_vars)

    print('\n***************************************************************',
          '\n************************** all done! **************************',
          '\n***************************************************************\n')

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main(sys.argv[1:])

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################