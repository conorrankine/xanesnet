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
from xanesnet.normalise import norm_xanes
from xanesnet.convolute import ArctanConvoluter
from xanesnet.descriptors import RDC
from xanesnet.descriptors import WACSF

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def learn(
    x_path: str,
    y_path: str,
    descriptor_type: str,
    descriptor_params: dict = {},
    kfold_params: dict = {},
    hyperparams: dict = {},
    max_samples: int = None,
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
        x_path (str): The path to the .xyz (X) data; expects a directory
            containing .xyz files.
        y_path (str): The path to the XANES spectral (Y) data; expects a
            directory containing .txt FDMNES output files.
        descriptor_type (str): The type of descriptor to use; the descriptor
            transforms molecular systems into fingerprint feature vectors
            that encodes the local environment around absorption sites.
            See xanesnet.descriptors for additional information.
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

    sample_ids = list(
        set(list_filestems(x_path)) & set(list_filestems(y_path))
    )

    sample_ids.sort()

    descriptors = {'rdc': RDC, 'wacsf': WACSF}
    descriptor = descriptors[descriptor_type](**descriptor_params)

    n_samples = len(sample_ids)
    n_x_features = descriptor.get_len()
    n_y_features = linecount(y_path / f'{sample_ids[0]}.txt') - 2

    x = np.full((n_samples, n_x_features), np.nan)
    print('>> preallocated {}x{} array for X data...'.format(*x.shape))
    #y_pn is the xanes spectra pre-normalised. y is used in the code.
    y_pn = np.full((n_samples, n_y_features), np.nan)
    y = np.full((n_samples, n_y_features), np.nan)
    print('>> preallocated {}x{} array for Y data...'.format(*y.shape))
    print('>> ...everything preallocated!\n')

    print('>> loading data into array(s)...')
    for i, sample_id in enumerate(tqdm.tqdm(sample_ids)):
        x[i,:] = descriptor.transform(
            load_xyz(x_path / f'{sample_id}.xyz')
        )
        e, y_pn[i,:] = load_xanes(y_path / f'{sample_id}.txt')
        y[i,:] = norm_xanes(y_pn[i,:])
    print('>> ...loaded into array(s) and normalised!\n')

    if save:
        model_dir = unique_path(Path('.'), 'model')
        model_dir.mkdir()
        with open(model_dir / 'descriptor.pickle', 'wb') as f:
            pickle.dump(descriptor, f)
        with open(model_dir / 'dataset.npz', 'wb') as f:
            np.savez_compressed(f, x = x, y = y, e = e)

    print('>> shuffling and selecting data...')
    shuffle(x, y, random_state = rng, n_samples = max_samples)
    print('>> ...shuffled and selected!\n')

    net = KerasRegressor(
        build_fn = build_mlp, 
        inp_dim = x[0].size, 
        out_dim = y[0].size, 
        **hyperparams,
        callbacks = set_callbacks(**callbacks),
        epochs = epochs,
        random_state = rng,
        verbose = 2
    )

    print('>> setting up preprocessing pipeline...')
    pipeline = Pipeline([('scaler', StandardScaler()), ('net', net)])
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

def predict(
    model_dir: str,
    x_path: str,
    conv_params: dict = {},
):
    """
    PREDICT. The model state is restored from a model directory containing
    serialised scaling/pipeline objects and the serialised model, .xyz (X)
    data are loaded and transformed, and the model is used to predict XANES
    spectral (Y) data; convolution of the Y data is also possible if
    {conv_params} are provided (see xanesnet/convolute.py).

    Args:
        model_dir (str): The path to a model.[?] directory created by
            the LEARN routine.
        x_path (str): The path to the .xyz (X) data; expects a directory
            containing .xyz files.
        conv_params (dict, optional): A dictionary of keyword arguments
            passed to the convoluter on initialisation; expects, at least,
            the absorption edge energy ('e_edge') and Fermi energy 
            ('e_fermi'). See xanesnet/convolute.py for additional info.
            Defaults to {}.
    """

    model_dir = Path(model_dir)

    x_path = Path(x_path)

    sample_ids = list_filestems(x_path)

    with open(model_dir / 'descriptor.pickle', 'rb') as f:
        descriptor = pickle.load(f)

    n_samples = len(sample_ids)
    n_x_features = descriptor.get_len()

    x = np.full((n_samples, n_x_features), np.nan)
    print('>> preallocated {}x{} array for X data...'.format(*x.shape))
    print('>> ...everything preallocated!\n')

    print('>> loading data into array(s)...')
    for i, sample_id in enumerate(tqdm.tqdm(sample_ids)):
        x[i,:] = descriptor.transform(
            load_xyz(x_path / f'{sample_id}.xyz')
        )
    print('>> ...loaded!\n')

    pipeline = load_pipeline(
        model_dir / 'net.keras',
        model_dir / 'pipeline.pickle'
    )

    print('>> predicting Y data with neural net...')
    y_predict = pipeline.predict(x)
    if y_predict.ndim == 1:
        y_predict = y_predict.reshape(-1, y_predict.size)
    print('>> ...predicted Y data!\n')

    predict_dir = unique_path(Path('.'), 'predictions')
    predict_dir.mkdir()

    with open(model_dir / 'dataset.npz', 'rb') as f:
        e = np.load(f)['e']

    print('>> saving Y data predictions...')
    for sample_id, y_predict_ in tqdm.tqdm(zip(sample_ids, y_predict)):
        save_path = predict_dir / f'{sample_id}.txt'
        save_xanes(save_path, e, y_predict_)
    print('...saved!\n')

    if conv_params:
        
        convoluter = ArctanConvoluter(**conv_params)

        print('>> saving convoluted Y data predictions...')
        for sample_id, y_predict_ in tqdm.tqdm(zip(sample_ids, y_predict)):
            save_path = predict_dir / f'{sample_id}_conv.txt'
            save_xanes(save_path, e, convoluter.convolute(e, y_predict_))
        print('...saved!\n')
        
    return 0
