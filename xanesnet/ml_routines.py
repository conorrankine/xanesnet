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
from typing import Union
from pathlib import Path
from . import utils
from numpy import save
from xanesnet.config import load_config
from xanesnet.dataset import load_dataset_from_data_src
from xanesnet.descriptors import RDC, WACSF
from xanesnet.xanes import XANESSpectrumTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def train(
    data_src: Union[Path, list[Path], list[Path, Path]],
    config: Path = None
):

    config = load_config(
        config if config is not None else 'test_config.yaml'
    )

    output_dir = utils.unique_path(Path.cwd(), 'xanesnet_output')
    if not output_dir.is_dir():
        output_dir.mkdir(parents = True)

    descriptors = {
        'rdc': RDC,
        'wacsf': WACSF
    }
   
    print(f'{config["descriptor"]["type"].upper()} parameters:')
    print('-' * 35)
    for key, val in config["descriptor"]["params"].items():
        print(f'{key:<25}{val:>10}')
    print('-' * 35 + '\n')

    descriptor = descriptors.get(config["descriptor"]["type"])(
        **config["descriptor"]["params"]
    )

    with open(output_dir / 'descriptor.pkl', 'wb') as f:
        pickle.dump(descriptor, f)

    print('spectrum preprocessing parameters:')
    print('-' * 35)
    for key, val in config["spectrum"]["params"].items():
        print(f'{key:<25}{val:>10}')
    print('-' * 35 + '\n')

    spectrum_transformer = XANESSpectrumTransformer(
        **config["spectrum"]["params"]
    )

    with open(output_dir / 'spectrum_transformer.pkl', 'wb') as f:
        pickle.dump(spectrum_transformer, f)

    print('loading + preprocessing data records from source...')
    x, y = load_dataset_from_data_src(
        *data_src,
        x_transformer = descriptor,
        y_transformer = spectrum_transformer
    )
    for data, data_src_ in zip((x, y), data_src):
        print(f'loaded {len(data)} records @ {data_src_}')

    for data, label in zip((x, y), ('x', 'y')):
        with open(output_dir / f'{label}.npy', 'wb') as f:
            save(f, data)

    print('neural network parameters:')
    print('-' * 35)
    for key, val in config["model"].items():
        if key != 'hidden_layer_sizes':
            print(f'{key:<25}{val:>10}')
    print('-' * 35 + '\n')    

    pipeline = Pipeline([
        ('feature_selection', VarianceThreshold(
            **config['feature_selection'])
        ),
        ('feature_scaling', StandardScaler(
            **config['feature_scaling'])
        ),
        ('model', MLPRegressor(
            **config['model'])
        )
    ])

    pipeline.fit(x, y)

    with open(output_dir / 'pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

def predict(
    data_src: Union[Path, list[Path]],
    model: Path
):

    pass