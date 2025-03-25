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
    """
    #TODO: complete docstring!

    Args:
        x_data_src (Path): _description_
        y_data_src (Path): _description_
        config (dict): _description_
    """
    
    random_state = RandomState(seed = config["random_state"]["seed"])

    output_dir = utils.unique_path(Path.cwd(), 'xanesnet_output')
    if not output_dir.is_dir():
        output_dir.mkdir(parents = True)

    _summarise_config_params(
        config,
        objects = [
            'descriptor',
            'spectrum_transformer',
            'model'
        ]
    )

    x_transformer, y_transformer, pipeline = _create_objects_from_config(
        config, random_state = random_state
    )

    print('\nloading + preprocessing data records from source...')
    x, y = load_dataset_from_data_src(
        x_data_src,
        y_data_src,
        x_transformer = x_transformer,
        y_transformer = y_transformer,
        verbose = True
    )
    for data, data_src in zip((x, y), (x_data_src, y_data_src)):
        print(f'loaded {len(data)} records @ {data_src}')

    for data, label in zip((x, y), ('x', 'y')):
        with open(output_dir / f'{label}.npy', 'wb') as f:
            save(f, data)

    pipeline.fit(x, y)

    _save_objects_to_model_dir(
        output_dir,
        {
            'descriptor': x_transformer,
            'spectrum_transformer': y_transformer,
            'pipeline': pipeline
        }
    )

    metric = get_metric(config["metric"]["type"].lower())

    score = metric(y, pipeline.predict(x))
    print(f'\nscore: {score:.6f} ({config["metric"]["type"].upper()})\n')

def validate(
    x_data_src: Path,
    y_data_src: Path,
    config: dict
):
    """
    #TODO: complete docstring!

    Args:
        x_data_src (Path): _description_
        y_data_src (Path): _description_
        config (dict): _description_
    """
    
    random_state = RandomState(seed = config["random_state"]["seed"])

    _summarise_config_params(
        config,
        objects = [
            'descriptor',
            'spectrum_transformer',
            'model'
        ]
    )

    x_transformer, y_transformer, pipeline = _create_objects_from_config(
        config, random_state = random_state
    )

    print('\nloading + preprocessing data records from source...')
    x, y = load_dataset_from_data_src(
        x_data_src,
        y_data_src,
        x_transformer = x_transformer,
        y_transformer = y_transformer,
        verbose = True
    )
    for data, data_src in zip((x, y), (x_data_src, y_data_src)):
        print(f'loaded {len(data)} records @ {data_src}')

    cv = RepeatedKFold(
        **config["kfold"], random_state = random_state
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
    """
    #TODO: complete docstring!

    Args:
        x_data_src (Path): _description_
        model (Path): _description_
    """
    
    output_dir = utils.unique_path(Path.cwd(), 'xanesnet_output')
    if not output_dir.is_dir():
        output_dir.mkdir(parents = True)

    x_transformer, y_transformer, pipeline = _load_objects_from_model_dir(
        model,
        objects = [
            'descriptor',
            'spectrum_transformer',
            'pipeline'
        ]
    )

    print('\nloading + preprocessing data records from source...')
    x, _ = load_dataset_from_data_src(
        x_data_src,
        x_transformer = x_transformer,
        verbose = True
    )
    print(f'...loaded {len(x)} records @ {x_data_src}')

    y_predicted = [
        XANES(y_transformer.energy_grid, y) for y in pipeline.predict(x)
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
    """
    #TODO: complete docstring!

    Args:
        x_data_src (Path): _description_
        y_data_src (Path): _description_
        model (Path): _description_
        metric_type (str, optional): _description_. Defaults to 'mse'.
    """
    
    x_transformer, y_transformer, pipeline = _load_objects_from_model_dir(
        model,
        objects = [
            'descriptor',
            'spectrum_transformer',
            'pipeline'
        ]
    )

    print('\nloading + preprocessing data records from source...')
    x, y = load_dataset_from_data_src(
        x_data_src,
        y_data_src,
        x_transformer = x_transformer,
        y_transformer = y_transformer,
        verbose = True
    )
    for data, data_src in zip((x, y), (x_data_src, y_data_src)):
        print(f'loaded {len(data)} records @ {data_src}')

    y_predicted = pipeline.predict(x)

    metric = get_metric(metric_type.lower())

    score = metric(y, y_predicted)
    print(f'\nscore: {score:.6f} ({metric_type.upper()})\n')
    
def _cross_validate(
    pipeline: Pipeline,
    x: ndarray,
    y: ndarray,
    cv: BaseCrossValidator,
    metric: callable
) -> dict:
    """
    #TODO: complete docstring!

    Args:
        pipeline (Pipeline): _description_
        x (ndarray): _description_
        y (ndarray): _description_
        cv (BaseCrossValidator): _description_
        metric (callable): _description_

    Returns:
        dict: _description_
    """
    
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
    """
    #TODO: complete docstring!

    Args:
        cv_results (dict): _description_
    """
    
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

def _summarise_config_params(
    config: dict,
    objects: list
):
    """
    #TODO: complete docstring!

    Args:
        config (dict): _description_
        objects (list): _description_
    """

    for object in objects:
        print(f'\n{object.replace("_", " ")} params:')
        utils.print_nested_dict(config[object])

def _create_objects_from_config(
    config: dict,
    random_state: RandomState
) -> tuple:
    """
    #TODO: complete docstring!

    Args:
        config (dict): _description_
        random_state (RandomState): _description_

    Returns:
        tuple: _description_
    """
    
    x_transformer = get_descriptor(
        config["descriptor"]["type"],
        params = config["descriptor"]["params"]
    )

    y_transformer = get_spectrum_transformer(
        config["spectrum_transformer"]["type"],
        params = config["spectrum_transformer"]["params"]
    )

    pipeline = Pipeline([
        ('feature_selection', VarianceThreshold(
            **config['feature_selection'])
        ),
        ('feature_scaling', StandardScaler(
            **config['feature_scaling'])
        ),
        ('model', MLPRegressor(
            **config['model']['params'], random_state = random_state)
        )
    ])

    return tuple(
        [x_transformer, y_transformer, pipeline]
    )

def _save_objects_to_model_dir(
    model_dir: Path,
    objects: dict
):
    """
    #TODO: complete docstring!

    Args:
        model_dir (Path): _description_
        objects (dict): _description_
    """
    
    for name, object in objects.items():
        with open(model_dir / f'{name}.pkl', 'wb') as f:
            pickle.dump(object, f)

def _load_objects_from_model_dir(
    model_dir: Path,
    objects: list
) -> tuple:
    """
    #TODO: complete docstring!

    Args:
        model_dir (Path): _description_
        objects (list): _description_

    Returns:
        tuple: _description_
    """
    
    object_files = [
        model_dir / f'{object}.pkl' for object in objects
    ]
    
    return tuple(
        [pickle.load(open(f, 'rb')) for f in object_files]
    )

def _write_predictions(
    y_predicted: list[XANES],
    output_dir: Path,
    output_filenames: list = None,
    format: str = 'csv'
):
    """
    #TODO: complete docstring!

    Args:
        y_predicted (list[XANES]): _description_
        output_dir (Path): _description_
        output_filenames (list, optional): _description_. Defaults to None.
        format (str, optional): _description_. Defaults to 'csv'.

    Raises:
        NotADirectoryError: _description_
        ValueError: _description_
    """
    
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
