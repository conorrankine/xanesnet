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

import numpy as np
from typing import Union, Optional, Callable
from pathlib import Path
from xanesnet.descriptors import _Descriptor
from xanesnet.xanes import XANESSpectrumTransformer

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def load_dataset_from_data_src(
    x_src: Path,
    y_src: Path = None,
    x_transformer: _Descriptor = None,
    y_transformer: XANESSpectrumTransformer = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Loads input (X) and target (Y) data from the specified source paths and
    applies preprocessing transforms if i) the specified source paths are
    directories and ii) X ()`x_transformer`) and Y (`y_transformer`)
    transformer objects that implement a `.transform()` method are passed; if
    the specified source paths point to .npz files, then these are unpacked and
    returned as-is without applying any preprocessing transforms.

    # TODO: extend documentation

    Args:
        x_src (Path): Source path for input (X) data.
        y_src (Path, optional): Source path for target (Y) data. Defaults to
            None.
        x_transformer (_Descriptor, optional): Transformer for the input (X)
            data; expects an instance of the `_Descriptor` class that has the
            `.transform()` method implemented. Defaults to None.
        y_transformer (XANESSpectrumTransformer, optional): Transformer for the
            target (Y) data; expects an instance of the
            `XANESSpectrumTransformer` class that has the `.transform()` method
            implemented. Defaults to None.

    Returns:
        tuple[np.ndarray, Optional[np.ndarray]]: Tuple of input (X) and
            (optionally) output (Y) data.
    """
    
    x = _load_from_data_src(
        x_src,
        x_transformer,
        load_x_data_from_dir        
    )

    y = _load_from_data_src(
        y_src,
        y_transformer,
        load_y_data_from_dir,
    ) if y_src else None

    return x, y

def _load_from_data_src(
    data_src: Path,
    data_transformer: Union[_Descriptor, XANESSpectrumTransformer],
    directory_data_loader: Callable
) -> np.ndarray:
    """
    Loads data from the specified source path and applies preprocessing
    transforms if i) the specified source path is a directory, and ii) a data
    transformer object that implements a `.transform()` method is passed; if
    the specified source path points to an .npz file, then this is unpacked
    and returned as-is without applying any preprocessing transforms.

    Args:
        data_src (Path): Source path for data.
        data_transformer (Union[_Descriptor, XANESSpectrumTransformer]):
            Transformer for the data; expects an instance of a class with the
            `.transform()` method implemented.
        directory_data_loader (Callable): Function for loading the data from
            a directory; the function is expected to take both the data source
            (`data_src`) and transformer (`data_transformer`) as arguments.

    Raises:
        FileNotFoundError: If `data_src` does not exist.
        ValueError: If `data_src` is not a valid/supported path.

    Returns:
        np.ndarray: Loaded data.
    """
    
    if data_src.is_file():
        return np.load(
            data_src
        )
    elif data_src.is_dir():
        return directory_data_loader(
            data_src,
            data_transformer
        )
    elif not data_src.exists():
        raise FileNotFoundError(
            f'{data_src} does not exist'
        )
    else:
        raise ValueError(
            f'{data_src} is not a valid/supported path'
        )
        
def load_x_data_from_dir(
    x_dir: Path,
    x_transformer: _Descriptor = None
) -> np.ndarray:
    
    pass

def load_y_data_from_dir(
    y_dir: Path,
    y_transformer: XANESSpectrumTransformer = None
) -> np.ndarray:
    
    pass
