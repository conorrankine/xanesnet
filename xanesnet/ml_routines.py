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
from pathlib import Path
from . import utils
from tqdm import tqdm
from numpy import ndarray, save
from numpy.random import RandomState
from xanesnet.dataset import load_dataset_from_data_src
from xanesnet.descriptors import get_descriptor
from xanesnet.xanes import (
    XANES, XANESSpectrumTransformer, get_spectrum_transformer, write
)
from xanesnet.metrics import get_metric
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def train(
    x_data_src: Path,
    y_data_src: Path,
    config: dict
):
    
    x, y, pipeline, output_dir = _setup_train(
        x_data_src,
        y_data_src,
        config,
        verbose = True
    )

    pipeline.fit(x, y)

    with open(output_dir / 'pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    metric = get_metric(config["metric"]["type"].lower())

    score = metric(y, pipeline.predict(x))
    print(f'\nscore: {score:.6f} ({config["metric"]["type"].upper()})\n')

def predict(
    x_data_src: Path,
    model: Path
):

    y_predicted, spectrum_transformer, output_dir = _setup_predict(
        x_data_src,
        model,
        verbose = True
    )

    output_filenames = [
        f'{file_stem}.csv' for file_stem in utils.list_file_stems(x_data_src)
    ] if x_data_src.is_dir() else None

    _write_predictions(
        y_predicted,
        spectrum_transformer,
        output_dir,
        output_filenames,
        verbose = True
    )

def _setup_train(
    x_data_src: Path,
    y_data_src: Path,
    config: dict,
    verbose: bool = False
) -> tuple[ndarray, ndarray, Pipeline, Path]:
    
    rng = RandomState(seed = config["random_state"]["seed"])

    output_dir = utils.unique_path(Path.cwd(), 'xanesnet_output')
    if not output_dir.is_dir():
        output_dir.mkdir(parents = True)
  
    if verbose:
        print(f'\n{config["descriptor"]["type"].upper()} parameters:')
        utils.print_nested_dict(
            config["descriptor"]["params"]
        )

    descriptor = get_descriptor(
        config["descriptor"]["type"],
        params = config["descriptor"]["params"]
    )

    with open(output_dir / 'descriptor.pkl', 'wb') as f:
        pickle.dump(descriptor, f)

    if verbose:
        print('\nspectrum preprocessing parameters:')
        utils.print_nested_dict(
            config["spectrum_transformer"]["params"]
        )

    spectrum_transformer = get_spectrum_transformer(
        config["spectrum_transformer"]["type"],
        params = config["spectrum_transformer"]["params"]
    )

    with open(output_dir / 'spectrum_transformer.pkl', 'wb') as f:
        pickle.dump(spectrum_transformer, f)

    if verbose:
        print('\nloading + preprocessing data records from source...')
    x, y = load_dataset_from_data_src(
        x_data_src,
        y_data_src,
        x_transformer = descriptor,
        y_transformer = spectrum_transformer,
        verbose = verbose
    )
    if verbose:
        for data, data_src in zip((x, y), (x_data_src, y_data_src)):
            print(f'loaded {len(data)} records @ {data_src}')

    for data, label in zip((x, y), ('x', 'y')):
        with open(output_dir / f'{label}.npy', 'wb') as f:
            save(f, data)

    if verbose:
        print('\nneural network parameters:')
        utils.print_nested_dict(
            config["model"]
        )

    pipeline = Pipeline([
        ('feature_selection', VarianceThreshold(
            **config['feature_selection'])
        ),
        ('feature_scaling', StandardScaler(
            **config['feature_scaling'])
        ),
        ('model', MLPRegressor(
            **config['model'], random_state = rng)
        )
    ])

    return x, y, pipeline, output_dir

def _setup_predict(
    x_data_src: Path,
    model: Path,
    verbose: bool = False
) -> tuple[ndarray, XANESSpectrumTransformer, Path]:

    output_dir = utils.unique_path(Path.cwd(), 'xanesnet_output')
    if not output_dir.is_dir():
        output_dir.mkdir(parents = True)

    with open(model / 'descriptor.pkl', 'rb') as f:
        descriptor = pickle.load(f)

    with open(model / 'spectrum_transformer.pkl', 'rb') as f:
        spectrum_transformer = pickle.load(f)

    if verbose:
        print('\nloading + preprocessing data records from source...')
    x, _ = load_dataset_from_data_src(
        x_data_src,
        x_transformer = descriptor,
        verbose = verbose
    )
    if verbose:
        print(f'...loaded {len(x)} records @ {x_data_src}')

    with open(model / 'pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    y_predicted = pipeline.predict(x)

    return y_predicted, spectrum_transformer, output_dir

def _write_predictions(
    y_predicted: list,
    spectrum_transformer: XANESSpectrumTransformer,
    output_dir: Path,
    output_filenames: list = None,
    format: str = 'csv',
    verbose: bool = False
):
    
    if not output_dir.is_dir():
        raise NotADirectoryError(
            f'{output_dir} does not exist or is not a directory'
        )
    
    if output_filenames and len(output_filenames) != len(y_predicted):
            raise ValueError(
                '`y_predicted` and `output_filenames` should have the same '
                'length'
            )

    if verbose:
        print('\noutputting predictions...')

    for i, y in tqdm(
        enumerate(y_predicted),
        total = len(y_predicted),
        ncols = 60,
        nrows = None,
        disable = False if verbose else True
    ):
        xanes = XANES(spectrum_transformer._e_aux, y, e0 = 0.0)
        output_filename = (
            output_filenames[i] if output_filenames else f'{i:06d}.{format}'
        ) 
        write(output_dir / output_filename, xanes, format = format)
    
    if verbose:
        print(f'...output {len(y_predicted)} predictions @ {output_dir}/\n')
