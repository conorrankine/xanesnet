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

import pickle
from typing import Union
from pathlib import Path
from . import utils
from xanesnet.config import load_config
from xanesnet.descriptors import RDC, WACSF

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def train(
    data_src: Union[Path, list[Path], list[Path, Path]],
    config: Path = None
):

    config = load_config(
        config if config is not None else 'test_config.yaml'
    )

    output_dir = utils.unique_path(Path.cwd(), 'xanesnet_output')
    if not output_dir.is_dir():
        output_dir.mkdir(parents = True)

    descriptors = {
        'rdc': RDC,
        'wacsf': WACSF
    }
   
    print(f'{config["descriptor"]["type"].upper()} parameters:')
    print('-' * 35)
    for key, val in config["descriptor"]["params"].items():
        print(f'{key:<25}{val:>10}')
    print('-' * 35 + '\n')

    descriptor = descriptors.get(config["descriptor"]["type"])(
        **config["descriptor"]["params"]
    )

    with open(output_dir / 'descriptor.pkl', 'wb') as f:
        pickle.dump(descriptor, f)

def predict(
    data_src: Union[Path, list[Path]],
    model: Path
):

    pass