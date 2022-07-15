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

import numpy as np
import pickle as pickle
import tqdm as tqdm

from pathlib import Path
from numpy.random import RandomState
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from xanesnet.io import load_xyz
from xanesnet.io import save_xyz
from xanesnet.io import load_xanes
from xanesnet.io import save_xanes
from xanesnet.io import load_pipeline
from xanesnet.io import save_pipeline
from xanesnet.dnn import check_gpu_support
from xanesnet.dnn import set_callbacks
from xanesnet.dnn import build_mlp
from xanesnet.utils import unique_path
from xanesnet.utils import linecount
from xanesnet.utils import list_filestems
from xanesnet.utils import print_cross_validation_scores
from xanesnet.descriptors import RDC
from xanesnet.descriptors import WACSF
from xanesnet.xanes import XANES

###############################################################################
################################ MAIN FUNCTION ################################
###############################################################################

def main(
    x_path: str,
    y_path: str,
    descriptor_type: str,
    descriptor_params: dict = {},
    kfold_params: dict = {},
    hyperparams: dict = {},
    max_samples: int = None,
    variance_threshold: float = 0.0,
    epochs: int = 100,
    callbacks: dict = {},
    seed: int = None,
    save: bool = True,
):
    """
    LEARN. The .xyz (X) and XANES spectral (Y) data are loaded and transformed;
    a neural network is set up and fit to these data to find an Y <- X mapping.
    K-fold cross-validation is possible if {kfold_params} are provided. 
    
    Args:
        x_path (str): The path to the .xyz (X) data; expects either a directory
            containing .xyz files or a .npz archive file containing an 'x' key,
            e.g. the `dataset.npz` file created when save == True. If a .npz
            archive is provided, save is toggled to False, and the data are not
            preprocessed, i.e. they are expected to be ready to be passed into
            the neural net.
        y_path (str): The path to the XANES spectral (Y) data; expects either a
            directory containing .txt FDMNES output files or a .npz archive
            file containing 'y' and 'e' keys, e.g. the `dataset.npz` file
            created when save == True. If a .npz archive is provided, save is
            toggled to False, and the data are not preprocessed, i.e. they are
            expected to be ready to be passed into the neural net.
        descriptor_type (str): The type of descriptor to use; the descriptor
            transforms molecular systems into fingerprint feature vectors
            that encodes the local environment around absorption sites. See
            xanesnet.descriptors for additional information.
        descriptor_params (dict, optional): A dictionary of keyword
            arguments passed to the descriptor on initialisation.
            Defaults to {}.
        kfold_params (dict, optional): A dictionary of keyword arguments
            passed to a scikit-learn K-fold splitter (KFold or RepeatedKFold).
            If an empty dictionary is passed, no K-fold splitting is carried
            out, and all available data are exposed to the neural network.
            Defaults to {}.
        hyperparams (dict, optional): A dictionary of hyperparameter
            definitions used to configure a Sequential Keras neural network.
            Defaults to {}.
        max_samples (int, optional): The maximum number of samples to select
            from the X/Y data; the samples are chosen according to a uniform
            distribution from the full X/Y dataset.
            Defaults to None.
        variance_threshold (float, optional): The minimum variance threshold
            tolerated for input features; input features with variances below
            the variance threshold are eliminated.
            Defaults to 0.0.
        epochs (int, optional): The maximum number of epochs/cycles.
            Defaults to 100.
        callbacks (dict, optional): A dictionary of keyword arguments passed
            to set up Keras neural network callbacks; each argument is
            expected to be dictionary of arguments for the defined callback,
            e.g. "earlystopping": {"patience": 10, "verbose": 1}
            Defaults to {}.
        seed (int, optional): A random seed used to initialise a Numpy
            RandomState random number generator; set the seed explicitly for
            reproducible results over repeated calls to the `learn` routine.
            Defaults to None.
        save (bool, optional): If True, a model directory (containing data,
            serialised scaling/pipeline objects, and the serialised model)
            is created; this is required to restore the model state later.
            Defaults to True.
    """

    rng = RandomState(seed = seed)

    x_path = Path(x_path)
    y_path = Path(y_path)
    
    for path in (x_path, y_path):
        if not path.exists():
            err_str = f'path to X/Y data ({path}) doesn\'t exist'
            raise FileNotFoundError(err_str)

    if x_path.is_dir() and y_path.is_dir():
        print('>> loading data from directories...\n')

        ids = list(
            set(list_filestems(x_path)) & set(list_filestems(y_path))
        )

        ids.sort()

        descriptors = {
            'rdc': RDC,
            'wacsf': WACSF
        }
        
        descriptor = (
            descriptors.get(descriptor_type)(**descriptor_params)
        )

        n_samples = len(ids)
        n_x_features = descriptor.get_len()
        n_y_features = linecount(y_path / f'{ids[0]}.txt') - 2

        x = np.full((n_samples, n_x_features), np.nan)
        print('>> preallocated {}x{} array for X data...'.format(*x.shape))
        y = np.full((n_samples, n_y_features), np.nan)
        print('>> preallocated {}x{} array for Y data...'.format(*y.shape))
        print('>> ...everything preallocated!\n')

        print('>> loading data into array(s)...')
        for i, id_ in enumerate(tqdm.tqdm(ids)):
            with open(x_path / f'{id_}.xyz', 'r') as f:
                atoms = load_xyz(f)
            x[i,:] = descriptor.transform(atoms)
            with open(y_path / f'{id_}.txt', 'r') as f:
                xanes = load_xanes(f)
            e, y[i,:] = xanes.spectrum
        print('>> ...loaded into array(s)!\n')

        if save:
            model_dir = unique_path(Path('.'), 'model')
            model_dir.mkdir()
            with open(model_dir / 'descriptor.pickle', 'wb') as f:
                pickle.dump(descriptor, f)
            with open(model_dir / 'dataset.npz', 'wb') as f:
                np.savez_compressed(f, ids = ids, x = x, y = y, e = e)

    elif x_path.is_file() and y_path.is_file():
        print('>> loading data from .npz archive(s)...\n')
        
        with open(x_path, 'rb') as f:
            x = np.load(f)['x']
        print('>> ...loaded {}x{} array of X data'.format(*x.shape))
        with open(y_path, 'rb') as f:
            y = np.load(f)['y']
            e = np.load(f)['e']
        print('>> ...loaded {}x{} array of Y data'.format(*y.shape))
        print('>> ...everything loaded!\n')

        if save:
            print('>> overriding save flag (running in `--no-save` mode)\n')
            save = False

    else:

        err_str = 'paths to X/Y data are expected to be either a) both ' \
            'files (.npz archives), or b) both directories'
        raise TypeError(err_str)

    print('>> shuffling and selecting data...')
    x, y = shuffle(x, y, random_state = rng, n_samples = max_samples)
    print('>> ...shuffled and selected!\n')

    net = KerasRegressor(
        build_fn = build_mlp, 
        out_dim = y[0].size, 
        **hyperparams,
        callbacks = set_callbacks(**callbacks),
        epochs = epochs,
        random_state = rng,
        verbose = 2
    )

    print('>> setting up preprocessing pipeline...')
    pipeline = Pipeline([
        ('variance_threshold', VarianceThreshold(variance_threshold)),
        ('scaler', StandardScaler()),
        ('net', net)
    ])
    for i, step in enumerate(pipeline.get_params()['steps']):
        print(f'  >> {i + 1}. ' + '{} :: {}'.format(*step))
    print('>> ...set up!\n')

    check_gpu_support()

    if kfold_params:

        kfold_spooler = RepeatedKFold(**kfold_params, random_state = rng)

        print('>> fitting neural net...')
        kfold_output = cross_validate(
            pipeline, 
            x, 
            y, 
            cv = kfold_spooler, 
            return_train_score = True,
            return_estimator = True, 
            verbose = kfold_spooler.get_n_splits()
        )
        print('...neural net fit!\n')

        print_cross_validation_scores(kfold_output)

        if save:
            for kfold_pipeline in kfold_output['estimator']:
                kfold_dir = unique_path(model_dir, 'kfold')
                save_pipeline(
                    kfold_dir / 'net.keras', 
                    kfold_dir / 'pipeline.pickle',
                    kfold_pipeline
                )

    else:

        print('>> fitting neural net...')
        pipeline.fit(x, y)
        print('>> ...neural net fit!\n')

        if save:
            save_pipeline(
                model_dir / f'net.keras', 
                model_dir / f'pipeline.pickle',
                pipeline
            )
    
    return 0
