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
from numpy import ndarray, save, load
from numpy.random import RandomState
from xanesnet.config import load_config
from xanesnet.dataset import load_dataset_from_data_src
from xanesnet.descriptors import RDC, WACSF
from xanesnet.xanes import XANES, XANESSpectrumTransformer, read, write
from xanesnet.metrics import mean_squared_error, mean_absolute_error
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
    config: Path = None
):

    x, y, pipeline, output_dir = _setup_train(
        x_data_src,
        y_data_src,
        config
    )

    pipeline.fit(x, y)

    with open(output_dir / 'pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    metrics = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error
    }

    metric = metrics.get(config["metric"]["type"])

    score = metric(y, pipeline.predict(x))
    print(
        f'\nfinal score: {score:.6f} ({config["metric"]["type"].upper()})\n'
    )

def predict(
    x_data_src: Path,
    model: Path
):

    with open(model / 'descriptor.pkl', 'rb') as f:
        descriptor = pickle.load(f)

    with open(model / 'spectrum_transformer.pkl', 'rb') as f:
        spectrum_transformer = pickle.load(f)

    print('\nloading + preprocessing data records from source...')
    x, _ = load_dataset_from_data_src(
        x_data_src,
        x_transformer = descriptor,
        verbose = True
    )
    print(f'...loaded {len(x)} records @ {x_data_src}')

    with open(model / 'pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    y_predicted = pipeline.predict(x)

    output_dir = utils.unique_path(Path.cwd(), 'xanesnet_output')
    if not output_dir.is_dir():
        output_dir.mkdir(parents = True)

    output_filenames = [
        f'{file_stem}.csv' for file_stem in utils.list_file_stems(x_data_src)
    ] if x_data_src.is_dir() else None

    _write_predictions(
        y_predicted,
        spectrum_transformer,
        output_dir,
        output_filenames,
        format = 'csv',
        verbose = True
    )

def _setup_train(
    x_data_src: Path,
    y_data_src: Path,
    config: Path = None
) -> tuple[ndarray, ndarray, Pipeline, Path]:

    config = load_config(
        config if config is not None else 'xanesnet_2021.yaml'
    )

    rng = RandomState(seed = config["random_state"]["seed"])

    output_dir = utils.unique_path(Path.cwd(), 'xanesnet_output')
    if not output_dir.is_dir():
        output_dir.mkdir(parents = True)

    descriptors = {
        'rdc': RDC,
        'wacsf': WACSF
    }
   
    print(f'\n{config["descriptor"]["type"].upper()} parameters:')
    utils.print_nested_dict(
        config["descriptor"]["params"]
    )

    descriptor = descriptors.get(config["descriptor"]["type"])(
        **config["descriptor"]["params"]
    )

    with open(output_dir / 'descriptor.pkl', 'wb') as f:
        pickle.dump(descriptor, f)

    print('\nspectrum preprocessing parameters:')
    utils.print_nested_dict(
        config["spectrum"]["params"]
    )

    spectrum_transformer = XANESSpectrumTransformer(
        **config["spectrum"]["params"]
    )

    with open(output_dir / 'spectrum_transformer.pkl', 'wb') as f:
        pickle.dump(spectrum_transformer, f)

    print('\nloading + preprocessing data records from source...')
    x, y = load_dataset_from_data_src(
        x_data_src,
        y_data_src,
        x_transformer = descriptor,
        y_transformer = spectrum_transformer,
        verbose = True
    )
    for data, data_src in zip((x, y), (x_data_src, y_data_src)):
        print(f'loaded {len(data)} records @ {data_src}')

    for data, label in zip((x, y), ('x', 'y')):
        with open(output_dir / f'{label}.npy', 'wb') as f:
            save(f, data)

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
