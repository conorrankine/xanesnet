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

import xanesnet as xn
import datetime
from argparse import ArgumentParser, Namespace
from pathlib import Path
from .config import load_config

###############################################################################
############################## ARGUMENT PARSING ###############################
###############################################################################

def parse_args() -> Namespace:
    """
    Parses command line arguments for `xanesnet:cli.py`.

    Returns:
        argparse.Namespace: Parsed command line arguments as an
        argparse.Namespace object that holds the arguments as attributes.
    """

    p = ArgumentParser()

    p.add_argument(
        '--version', '-v', action = 'version', version = xn.__version__
    )
    
    sub_p = p.add_subparsers(
        dest = 'mode'
    )

    train_p = sub_p.add_parser(
        'train',
        help = 'train a model'                       
    )
    train_p.add_argument(
        'x_data_src', type = Path, 
        help = 'path to the input (X) data source'
    )
    train_p.add_argument(
        'y_data_src', type = Path, 
        help = 'path to the output/target (Y) data source'
    )    
    train_p.add_argument(
        '-c', '--config', type = Path, default = None,
        help = 'path to a .yaml configurational file'
    )

    validate_p = sub_p.add_parser(
        'validate',
        help = '(cross-)validate a model'                       
    )
    validate_p.add_argument(
        'x_data_src', type = Path, 
        help = 'path to the input (X) data source'
    )
    validate_p.add_argument(
        'y_data_src', type = Path, 
        help = 'path to the output/target (Y) data source'
    )    
    validate_p.add_argument(
        '-c', '--config', type = Path, default = None,
        help = 'path to a .yaml configurational file'
    )

    predict_p = sub_p.add_parser(
        'predict',
        help = 'make predictions using your trained model'
    )
    predict_p.add_argument(
        'x_data_src', type = Path,
        help = 'path to the input (X) data source'
    )
    predict_p.add_argument(
        'model', type = Path,
        help = 'path to the trained model'
    )

    evaluate_p = sub_p.add_parser(
        'evaluate',
        help = 'evaluate your trained model against a metric'
    )
    evaluate_p.add_argument(
        'x_data_src', type = Path, 
        help = 'path to the input (X) data source'
    )
    evaluate_p.add_argument(
        'y_data_src', type = Path, 
        help = 'path to the output/target (Y) data source'
    )
    evaluate_p.add_argument(
        'model', type = Path,
        help = 'path to the trained model'
    )
    evaluate_p.add_argument(
        '-m', '--metric', type = str, default = 'mse',
        help = 'metric to use for model evaluation'
    )

    args = p.parse_args()

    return args  

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

def main():

    datetime_ = datetime.datetime.now()
    print(f'launched @ {datetime_.strftime("%H:%M:%S (%Y-%m-%d)")}')

    args = parse_args()

    header_f = Path(__file__).parent / 'assets' / 'banners' / 'banner.txt'
    with open(header_f, 'r') as f:
        for line in f.readlines():
            print(line.rstrip())
    print('\n')

    if args.mode in ('train', 'validate'):
        config = load_config(
            args.config if args.config is not None else 'xanesnet_2021.yaml'
        )

    if args.mode == 'train':
        xn.train(
            args.x_data_src,
            args.y_data_src,
            config = config
        )
    if args.mode == 'validate':
        xn.validate(
            args.x_data_src,
            args.y_data_src,
            config = config
        )
    if args.mode == 'predict':
        xn.predict(
            args.x_data_src,
            args.model
        )
    if args.mode == 'evaluate':
        xn.evaluate(
            args.x_data_src,
            args.y_data_src,
            args.model,
            args.metric
        )

    datetime_ = datetime.datetime.now()
    print(f'finished @ {datetime_.strftime("%H:%M:%S (%Y-%m-%d)")}')

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################

if __name__ == '__main__':
    main()

################################################################################
############################### PROGRAM ENDS HERE ##############################
################################################################################