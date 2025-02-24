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
import numpy as np
import ase
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
    """
    Loads input (X) data from a directory of .xyz files; .xyz files are loaded
    as ase.Atoms objects by the `ase.io.read()` function and transformed into
    molecular descriptors by a transformer (`x_transformer`) that is expected
    to be an instance of a `_Descriptor` class; a 2D Numpy (`np.ndarray`) array
    is returned where each row corresponds to the transformed 1D Numpy array
    representation of each file in the directory.

    Args:
        x_dir (Path): Source path for the input (X) data directory.
        x_transformer (_Descriptor, optional): Transformer for the input (X)
            data; expects an instance of the `_Descriptor` class. Defaults to
            None.

    Returns:
        np.ndarray: Loaded input (X) data.
    """

    x = _load_data_from_dir(
        x_dir,
        data_transformer = x_transformer,
        file_data_loader = ase.io.read
    )
    
    return x

def load_y_data_from_dir(
    y_dir: Path,
    y_transformer: XANESSpectrumTransformer = None
) -> np.ndarray:
    """
    Loads target (Y) data from a directory of XANES spectrum files; XANES
    spectrum files are loaded as xanesnet.XANES objects by the 
    `xanesnet.xanes.read()` function and transformed (e.g. shifted, scaled,
    etc.) by a transformer (`y_transformer`) that is expected to be an
    instance of the `XANESSpectrumTransformer` class; a 2D Numpy (`np.ndarray`)
    array is returned where each row corresponds to the transformed 1D Numpy
    array representation of each file in the directory.

    Args:
        y_dir (Path): Source path for the target (Y) data directory.
        y_transformer (XANESSpectrumTransformer, optional): Transformer for the
        target (Y) data; expects an instance of the `XANESSpectrumTransformer`
        class. Defaults to None.

    Returns:
        np.ndarray: Loaded target (Y) data.
    """

    y = _load_data_from_dir(
        y_dir,
        data_transformer = y_transformer,
        file_data_loader = xn.xanes.read
    )

    return y

def _load_data_from_dir(
    data_dir: Path,
    data_transformer: Union[_Descriptor, XANESSpectrumTransformer],
    file_data_loader: Callable
) -> np.ndarray:
    """
    Loads data from a directory of files; files are loaded as appropriate
    objects by the `file_data_loader` function and transformed by a
    transformer `data_transformer` that implements the `.transform()` method
    into 1D Numpy (`np.ndarray) arrays; a 2D Numpy array is returned where
    each row corresponds to the transformed 1D Numpy array representation of
    each file in the directory. 

    Args:
        data_dir (Path): Source path for the data directory.
        data_transformer (Union[_Descriptor, XANESSpectrumTransformer]):
            Transformer for the data; expects that  `.transform()` method is
            implemented.
        file_data_loader (Callable): Function for loading files as objects that
            `data_transformer` can work with via the `.transform()` method.

    Returns:
        np.ndarray: Loaded data.
    """
    
    if data_transformer is None:
        raise NotImplementedError(
            'returning data as a list of file-loaded objects (i.e. without '
            'transformation into 1D Numpy arrays) is not yet supported'
        )
    else:
        n_samples = sum(1 for f in data_dir.iterdir() if f.is_file())
        if n_samples == 0:
            raise FileNotFoundError(
                f'no files found @ {data_dir}'
            )
        n_features = data_transformer.get_len()
        if n_features == 0:
            raise ValueError(
                'check data transformer configuration; cannot create '
                'zero-length feature vectors'
            )
        data = np.full((n_samples, n_features), np.nan)
        for i, f in enumerate(data_dir.iterdir()):
            if f.is_file():
                try:
                    data[i,:] = data_transformer.transform(
                        file_data_loader(f)
                    )
                except Exception:
                    print(f'error encountered processing file: {f}')
                    raise

    return data
