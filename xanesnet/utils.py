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

from pathlib import Path
from typing import Iterable

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def unique_path(path: Path, base_name: str) -> Path:
    # returns a unique path from `p`/`base_name`_001, `p`/`base_name`_002,
    # `p`/`base_name`_003, etc.

    n = 0
    while True:
        n += 1
        unique_path = path / (base_name + f'_{n:03d}')
        if not unique_path.exists():
            return unique_path
        
def print_nested_dict(
    dict_: dict,
    key_width: int = 35,
    val_width: int = 25,
    indent: int = 0
) -> None:
    """
    Prints a formatted table of `key`:`val` pairs contained in a (nested)
    dictionary, exploring (nested) subdictionaries recursively and printing
    the `key`:`val` pairs contained within with key indentation.

    Args:
        dict_ (dict): Dictionary.
        key_width (int, optional): Width of the column containing dictionary
            keys (`key`) for printout. Defaults to 35.
        val_width (int, optional): Width of the column containing dictionary
            values (`val`) for printout. Defaults to 25.
        indent (int, optional): Indentation level. Defaults to 0.

    Raises:
        TypeError: If a `key`:`val` pair has a value of an unsupported type.
    """
    
    indent_str = '  ' * indent
    key_width = max(10, key_width - (2 * indent))

    if indent == 0:
        print('-' * (key_width + val_width))
    
    for key, val in dict_.items():
        try:
            print(
                f'{indent_str}{key:<{key_width}}{val:>{val_width}}'
            )
        except TypeError:
            if isinstance(val, (list, tuple)):
                val = _iterable_to_str(val)
                print(
                    f'{indent_str}{key:<{key_width}}{val:>{val_width}}'
                    )
            elif isinstance(val, dict):
                print(
                    f'{indent_str}{key:<{key_width}}'
                )
                print_nested_dict(
                    val,
                    key_width = key_width,
                    val_width = val_width,
                    indent = indent + 1
                )
            else:
                raise TypeError(
                    f'can\'t handle value of type {type(val)} for key `{key}`'
                )

    if indent == 0:
        print('-' * (key_width + val_width))

def _iterable_to_str(
    iterable: Iterable
) -> str:
    """
    Converts an iterable (e.g., list, tuple, etc.) to a comma-separated string.

    Args:
        iterable (Iterable): Iterable to convert to a comma-separated string.

    Returns:
        str: Comma-separated string with each of the items in the iterable.
    """
    
    return ', '.join(str(item) for item in iterable)
