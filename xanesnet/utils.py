"""
XANESNET-REDUX
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

def unique_path(
    path: Path,
    prefix: str,
    extension: str = None
) -> Path:
    """
    Returns a unique file/directory path for a specified parent directory;
    `number` starts at `001` and increments by 1 until a unique file/directory
    path of the type `path`/`prefix`_`number`.`extension` is found.

    Args:
        path (Path): Path to the parent directory.
        prefix (str): Prefix for the unique path.
        extension (str): Extension for the unique path; defaults to `None`.

    Returns:
        Path: Unique file/directory path that does not already exist in the
            parent directory.
    """

    extension = f'.{extension}' if extension else ''

    n = 1
    while True:
        unique_path = path / (prefix + f'_{n:03d}' + extension) 
        if not unique_path.exists():
            return unique_path
        n += 1
        
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
    if indent >= 1:
        key_width = max(10, key_width - 2)

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

def list_files(
    path: Path
) -> list[Path]:
    """
    Returns a sorted list of files found in the specified directory.

    Args:
        path (Path): Path to a directory.

    Returns:
        list[Path]: List of files found in the specified directory.
    """

    if not path.is_dir():
        raise FileNotFoundError(
            f'{path} does not exist'
        )
    
    return sorted(
        file_ for file_ in path.iterdir() if file_.is_file()
    )

def list_file_stems(
    path: Path
) -> list[str]:
    """
    Returns a sorted list of file stems for files found in the specified
    directory.

    Args:
        path (Path): Path to a directory.

    Returns:
        list[str]: List of file stems for files found in the specified
            directory.
    """
    
    if not path.is_dir():
        raise FileNotFoundError(
            f'{path} does not exist'
        )
    
    return sorted(
        file_.stem for file_ in path.iterdir() if file_.is_file()
    )
