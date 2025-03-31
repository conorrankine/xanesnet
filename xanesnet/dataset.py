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
from tqdm import tqdm
from ase import io
from pathlib import Path
from typing import Optional, Callable, Any
from abc import ABC, abstractmethod

###############################################################################
################################## CLASSES ####################################
###############################################################################

class BaseTransformer(ABC):
    """
    An abstract base class for transformers used in, e.g., `load_dataset_from_
    data_src()` to transform input objects into (1D) np.ndarrays.


    This class defines a template for compatible transformers; subclasses are
    expected to implement the `.transform()` method to carry out the
    object -> (1D) np.ndarray transformation (as the exact transformation will
    be object- and implementation-specific) and the `.size` property to provide
    advance access to the number of elements in the (1D) np.ndarray
    representation returned by the `.transform()` method.
    """

    def __init__(self):
        
        pass

    @abstractmethod
    def transform(
        self,
        obj: Any
    ) -> np.ndarray:
        """
        Transforms an input object into a (1D) np.ndarray representation; the
        exact transformation will be object- and implementation-specific. 

        Args:
            obj (Any): Object to transform.

        Returns:
            np.ndarray: (1D) np.ndarray representation.
        """
        
        pass 

    @property
    @abstractmethod
    def size(
        self
    ) -> int:
        """
        Returns:
            int: Number of elements in the (1D) np.ndarray representation
                returned by the `.transform()` method.
        """

        pass

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def load_dataset_from_data_src(
    x_src: Path,
    y_src: Path = None,
    x_transformer: 'BaseTransformer' = None,
    y_transformer: 'BaseTransformer' = None,
    verbose: bool = False
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Loads input (X) and target (Y) data from the specified source paths: if
    a specified source path is a .npz file, data are unpacked and returned
    as-is; if a specified source path is a directory, files are loaded and
    transformed into np.ndarrays by the `.transform()` method of the
    `x_transformer` or `y_transformer` object (as appropriate; i.e. for
    `x_src` and `y_src`, respectively).

    Args:
        x_src (Path): Source path for input (X) data.
        y_src (Path, optional): Source path for target (Y) data. Defaults to
            None.
        x_transformer (BaseTransformer, optional): Transformer for the input
            (X) data; a subclass of `BaseTransformer`. Defaults to None.
        y_transformer (BaseTransformer, optional): Transformer for the target
            (Y) data; a subclass of `BaseTransformer`. Defaults to None.
        verbose (bool, optional): If `True`, and the data source(s) is/are
            a directory/directories, the data source(s) are printed and the
            data are loaded with a progress bar. Defaults to `False`.

    Raises:
        ValueError: If a different number of records are loaded from `x_src`
            and `y_src` (i.e. len(`x`) != len(`y`)) when `y_src` is not None.

    Returns:
        tuple[np.ndarray, Optional[np.ndarray]]: Tuple of input (X) and
            (optionally) output (Y) data (if `y_src` is not None).
    """
    
    x = _load_from_data_src(
        x_src,
        data_transformer = x_transformer,
        directory_data_loader = load_x_data_from_dir,
        verbose = verbose        
    )

    y = _load_from_data_src(
        y_src,
        data_transformer = y_transformer,
        directory_data_loader = load_y_data_from_dir,
        verbose = verbose
    ) if y_src else None

    if (y is not None) and not (len(x) == len(y)):
        raise ValueError(
            f'loaded a different number of records from `x_src` ({len(x)}) '
            f'and `y_src` ({len(y)}); double-check your data sources'
        )
    else:
        return x, y

def _load_from_data_src(
    data_src: Path,
    data_loader: Callable,
    data_transformer: 'BaseTransformer',
    verbose: bool = False
) -> np.ndarray:
    """
    Loads data from the specified source path: if `data_src` is a .npz file,
    data are unpacked and returned as-is; if `data_src` is a directory, files
    are loaded by the `directory_data_loader()` function and transformed into
    np.ndarrays by the `.transform()` method of the `data_transformer` object.

    Args:
        data_src (Path): Source path for data.
        data_transformer (BaseTransformer): Transformer for the data; a
            subclass of `BaseTransformer`.
        data_loader (Callable): Callable for loading the data files from a
            directory; the callable is expected to take both the data source
            (`data_src`) and transformer (`data_transformer`) as arguments.
        verbose (bool, optional): If `True`, and the data source is a
            directory, the data source is printed and the data are loaded with
            a progress bar. Defaults to `False`.

    Raises:
        FileNotFoundError: If `data_src` does not exist.
        ValueError: If `data_src` is not a valid/supported path.

    Returns:
        np.ndarray: Loaded data; the contents of the .npz file if `data_src`
            is a file, else a 2D array where each row corresponds to the data
            from each file in the directory if `data_src` is a directory.
    """
    
    if data_src.is_file():
        return np.load(
            data_src
        )
    elif data_src.is_dir():
        return data_loader(
            data_src,
            data_transformer,
            verbose = verbose
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
    x_transformer: 'BaseTransformer' = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Loads input (X) data from a directory of Cartesian coordinate (.xyz) files;
    files are loaded as `Atoms` objects by the `ase.io.read()` function and
    transformed into np.ndarrays representing their molecular feature vectors
    by the `.transform()` method of the `x_transformer` object (expected to be
    an instance of the `Descriptor` class).
     
    Args:
        x_dir (Path): Source path for the input (X) data directory.
        x_transformer (BaseTransformer, optional): Transformer for the input
            (X) data; a subclass of `BaseTransformer`. Defaults to None.
        verbose (bool, optional): If `True`, the data source is printed and the
            data are loaded with a progress bar. Defaults to `False`.

    Returns:
        np.ndarray: Loaded input (X) data; 2D array where each row corresponds
            to the data from each file in the directory.
    """

    x = _load_data_from_dir(
        x_dir,
        data_transformer = x_transformer,
        file_data_loader = io.read,
        verbose = verbose
    )
    
    return x

def load_y_data_from_dir(
    y_dir: Path,
    y_transformer: 'BaseTransformer' = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Loads target (Y) data from a directory of XANES spectrum files; files are
    are loaded as `XANES` objects by the `xanesnet.xanes.read()` function and
    transformed (with, e.g., shifting, scaling, convoluting, etc.) into
    np.ndarrays by the `.transform()` method of the `y_transformer` object
    (expected to be an instance of the `XANESSpectrumTransformer` class).
    
    Args:
        y_dir (Path): Source path for the target (Y) data directory.
        y_transformer (BaseTransformer, optional): Transformer for the target
            (Y) data; a subclass of `BaseTransformer`. Defaults to None.
        verbose (bool, optional): If `True`, the data source is printed and the
            data are loaded with a progress bar. Defaults to `False`.

    Returns:
        np.ndarray: Loaded target (Y) data; 2D array where each row corresponds
            to the data from each file in the directory.
    """

    y = _load_data_from_dir(
        y_dir,
        data_transformer = y_transformer,
        file_data_loader = xn.xanes.read,
        verbose = verbose
    )

    return y

def _load_data_from_dir(
    data_dir: Path,
    data_loader: Callable,
    data_transformer: 'BaseTransformer',
    verbose: bool = False
) -> np.ndarray:
    """
    Loads data from a directory of files; files are loaded as objects by the
    `file_data_loader` function and transformed into np.ndarrays by the
    `.transform()` method of the `data_transformer` object.  
    
    Args:
        data_dir (Path): Source path for the data directory.
        data_transformer (BaseTransformer): Transformer for the data; a
            subclass of `BaseTransformer`.
        data_loader (Callable): Callable for loading a data file; the callable
            is expected to take the data source (`data_src`) as an argument
            and return a compatible object for the `.transform()` method of
            the transformer (`data_transformer`).
        verbose (bool, optional): If `True`, the data source is printed and the
            data are loaded with a progress bar. Defaults to `False`.

    Raises:
        FileNotFoundError: If `data_dir` is an empty directory.
        ValueError: If the `.size` property of `data_transformer` is 0; i.e.
            if applying the `.transform()` method would result in zero-length
            feature/target vectors.

    Returns:
        np.ndarray: Loaded data; 2D array where each row corresponds to the
            data from each file in the directory.
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
        n_features = data_transformer.size
        if n_features == 0:
            raise ValueError(
                'check data transformer configuration; cannot create '
                'zero-length feature/target vectors'
            )
        data = np.full((n_samples, n_features), np.nan)
        if verbose:
            print(f'{data_dir}:')
        for i, f in tqdm(
            enumerate(sorted(data_dir.iterdir())),
            total = n_samples,
            ncols = 60,
            nrows = None,
            disable = False if verbose else True
        ):
            if f.is_file():
                try:
                    data[i,:] = data_transformer.transform(
                        data_loader(f)
                    )
                except Exception:
                    print(f'error encountered processing file: {f}')
                    raise

    return data
