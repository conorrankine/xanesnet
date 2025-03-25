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
from time import time
from tqdm import tqdm
from numpy import ndarray, save
from numpy.random import RandomState
from xanesnet.dataset import load_dataset_from_data_src
from xanesnet.descriptors import get_descriptor
from xanesnet.xanes import XANES, get_spectrum_transformer, write
from xanesnet.metrics import get_metric
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def train(
    x_data_src: Path,
    y_data_src: Path,
    config: dict
):
       
    x, y, pipeline, output_dir, _ = _setup_train(
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

def validate(
    x_data_src: Path,
    y_data_src: Path,
    config: dict
):

    x, y, pipeline, _, rng = _setup_train(
        x_data_src,
        y_data_src,
        config,
        verbose = True
    )

    cv = RepeatedKFold(
        **config["kfold"], random_state = rng
    )

    metric = get_metric(config["metric"]["type"].lower())

    print('\nsplitting data and cross-validating the pipeline...')
    cv_results = _cross_validate(
        pipeline,
        x,
        y,
        cv = cv,
        metric = metric
    )
    _print_cross_validation_results(cv_results)
    print(f'completed {cv.get_n_splits()} cross-validation cycles\n')

def predict(
    x_data_src: Path,
    model: Path
):
    
    output_dir = utils.unique_path(Path.cwd(), 'xanesnet_output')
    if not output_dir.is_dir():
        output_dir.mkdir(parents = True)

    descriptor, spectrum_transformer, pipeline = _load_components_from_model_dir(
        model
    )

    print('\nloading + preprocessing data records from source...')
    x, _ = load_dataset_from_data_src(
        x_data_src,
        x_transformer = descriptor,
        verbose = True
    )
    print(f'...loaded {len(x)} records @ {x_data_src}')

    y_predicted = [
        XANES(spectrum_transformer.energy_grid, y) for y in pipeline.predict(x)
    ]

    output_filenames = [
        f'{file_stem}.csv' for file_stem in utils.list_file_stems(x_data_src)
    ] if x_data_src.is_dir() else None

    print('\noutputting predictions as .csv files...')
    _write_predictions(
        y_predicted,
        output_dir,
        output_filenames
    )
    print(f'...output {len(y_predicted)} predictions @ {output_dir}/\n')

def evaluate(
    x_data_src: Path,
    y_data_src: Path,
    model: Path,
    metric_type: str = 'mse'
):
    
    descriptor, spectrum_transformer, pipeline = _load_components_from_model_dir(
        model
    )

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

    y_predicted = pipeline.predict(x)

    metric = get_metric(metric_type.lower())

    score = metric(y, y_predicted)
    print(f'\nscore: {score:.6f} ({metric_type.upper()})\n')
    
def _setup_train(
    x_data_src: Path,
    y_data_src: Path,
    config: dict,
    verbose: bool = False
) -> tuple[ndarray, ndarray, Pipeline, Path, RandomState]:
    
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

    return x, y, pipeline, output_dir, rng

def _cross_validate(
    pipeline: Pipeline,
    x: ndarray,
    y: ndarray,
    cv: BaseCrossValidator,
    metric: callable
) -> dict:
    
    cv_results = {}

    for i, (train_idx, valid_idx) in tqdm(
        enumerate(cv.split(x, y)), total = cv.get_n_splits(), ncols = 60
    ):
        
        x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        start_time = time()
        pipeline.fit(x_train, y_train)
        elapsed_time = time() - start_time
        
        cv_results.update(
            {f'fold_{i+1}': tuple([
                metric(pipeline.predict(x_train), y_train),
                metric(pipeline.predict(x_valid), y_valid),
                elapsed_time
            ])}
        )

    return cv_results

def _print_cross_validation_results(
    cv_results: dict
):
    
    print('-' * 60)
    print(f'{"fold":<10}{"train":>20}{"valid":>20}{"time":>10}')
    print('-' * 60)
    for key, val in cv_results.items():
        train_score, valid_score, time = val
        print(
            f'{key:<10}'
            f'{train_score:>20.6f}'
            f'{valid_score:>20.6f}'
            f'{time:>9.1f}s'
        )
    print('-' * 60)

def _load_components_from_model_dir(
    model_dir: Path
) -> tuple:
    
    component_files = [
        'descriptor.pkl',
        'spectrum_transformer.pkl',
        'pipeline.pkl'
    ]
    
    return tuple(
        [pickle.load(open(model_dir / f, 'rb')) for f in component_files]
    )

def _write_predictions(
    y_predicted: list[XANES],
    output_dir: Path,
    output_filenames: list = None,
    format: str = 'csv'
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

    for i, y_predicted_ in tqdm(
        enumerate(y_predicted), total = len(y_predicted), ncols = 60
    ):
        output_filename = (
            output_filenames[i] if output_filenames else f'{i:06d}.{format}'
        ) 
        write(output_dir / output_filename, y_predicted_, format = format)
