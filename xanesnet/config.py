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

import yaml
from pathlib import Path
from typing import Union
from importlib import resources
from . import configs

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def load_config(config_f: Union[str, Path]) -> dict:
    """
    Loads a .yaml configuration file (`config_f`). An attempt is made to load
    the .yaml file from the filesystem; if it is not found, an attempt is made
    to load the .yaml file as a packaged resource from `xanesnet:configs`.
    In the event that both attempts fail, a `FileNotFoundError` is raised.


    Args:
        config_f (Union[str, Path]): Path to a .yaml configuration file.

    Raises:
        FileNotFoundError: If the .yaml configuration file is not found.

    Returns:
        dict: Parsed .yaml configuration file as a dictionary.
    """

    config_f = Path(config_f)

    if config_f.is_file():
        with config_f.open() as f:
            return yaml.safe_load(f)

    try:
        with resources.open_text(configs, config_f.name) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:          
        raise FileNotFoundError(
            f'{config_f} was not found and is not an available configuration file '
            'through xanesnet:configs'
        ) from None
