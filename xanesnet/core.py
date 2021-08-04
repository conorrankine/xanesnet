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
import time as time

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from xanesnet.io import load_data_ids
from xanesnet.io import load_xyz
from xanesnet.io import save_xyz
from xanesnet.io import load_xanes
from xanesnet.io import save_xanes
from xanesnet.io import load_pipeline
from xanesnet.io import save_pipeline
from xanesnet.dnn import check_gpu_support
from xanesnet.dnn import set_callbacks
from xanesnet.dnn import build_mlp
from xanesnet.utils import print_cross_validation_scores
from xanesnet.convolute import ArctanConvoluter
from xanesnet.scalers import CentreScaler
from xanesnet.scalers import GroupMaxAbsScaler
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
    epochs: int = 100,
    callbacks: dict = {},
    **kwargs
):
    """
    LEARN. The .xyz (X) and XANES spectral (Y) data are loaded and transformed;
    a neural network is set up and fit to these data to find an Y <- X mapping.
    K-fold cross-validation is possible if {kfold_params} are provided.
    This function creates a model.[?] directory in the current workspace: 
    
    > model.[?]
      > objects
        > tf (contains the optimised Keras neural network(s), <model.keras>)
        > np (contains serialised .npy arrays)
        > sk (contains serialised .pickle preprocessors and pipelines)

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
        epochs (int, optional): The maximum number of epochs/cycles.
            Defaults to 100.
        callbacks (dict, optional): A dictionary of keyword arguments passed
            to set up Keras neural network callbacks; each argument is
            expected to be dictionary of arguments for the defined callback,
            e.g. "earlystopping": {"patience": 10, "verbose": 1}
            Defaults to {}.
    """

    model_dir = Path(f'./model.{int(time.time())}')
    obj_dir = model_dir / 'objects'
    np_dir = obj_dir / 'np'
    tf_dir = obj_dir / 'tf'
    sk_dir = obj_dir / 'sk'

    for d in (model_dir, obj_dir, np_dir, tf_dir, sk_dir):
        d.mkdir()

    if descriptor_type.lower() == 'rdc':
        descriptor = RDC(**descriptor_params)
    elif descriptor_type.lower() == 'wacsf':
        descriptor = WACSF(**descriptor_params)
    else:
        raise ValueError(f'descriptor type not recognised; ',
            'got {descriptor_type}')

    with open(sk_dir / 'descriptor.pickle', 'wb') as f:
        pickle.dump(descriptor, f)    

    x_path = Path(x_path)
    y_path = Path(y_path)

    ids = load_data_ids(x_path, y_path)

    x_spooler = (x_path / (id_ + '.xyz') for id_ in ids)
    print('>> loading X data...')
    x = [descriptor.transform(load_xyz(f)) for f in tqdm.tqdm(x_spooler)]
    print()

    y_spooler = (y_path / (id_ + '.txt') for id_ in ids)
    print('>> loading Y data...')
    e, y = zip(*[load_xanes(f) for f in tqdm.tqdm(y_spooler)])
    print()

    e = np.array(e, dtype = 'float32')
    if np.allclose(e[0], e):
        e = e[0]
    else:
        raise ValueError('Y data are not defined over a common energy ',
            'window; check .txt FDMNES output files for consistency')

    x = np.array(x, dtype = 'float32')
    y = np.array(y, dtype = 'float32')
 
    for array_name, array in {'x': x, 'y': y, 'e': e}.items():
        with open(np_dir / f'{array_name}.npy', 'wb') as f:
            np.save(f, array)

    scalers = [('centering', CentreScaler())]
    if isinstance(descriptor, RDC):
        scalers.append(('scaling', 
            GroupMaxAbsScaler()))
    elif isinstance(descriptor, WACSF):
        group_idx = (0,1)
        scalers.append(('g1_scaling', 
            GroupMaxAbsScaler(group_idx = group_idx)))
        if descriptor.use_g2:
            group_idx = (1,1+len(descriptor.g2_h))
            scalers.append(('g2_scaling', 
                GroupMaxAbsScaler(group_idx = group_idx)))
        if descriptor.use_g4:
            group_idx = (1+len(descriptor.g2_h),
                1+len(descriptor.g2_h)+len(descriptor.g4_h))
            scalers.append(('g4_scaling',
                GroupMaxAbsScaler(group_idx = group_idx)))

    net = KerasRegressor(
        build_fn = build_mlp, 
        inp_dim = x[0].size, 
        out_dim = y[0].size, 
        **hyperparams,
        callbacks = set_callbacks(**callbacks),
        epochs = epochs,
        verbose = 2
    )

    pipeline = Pipeline([*scalers, ('net', net)])

    if kfold_params:

        kfold_spooler = (RepeatedKFold(**kfold_params) 
            if 'n_repeats' in kfold_params else KFold(**kfold_params))

        check_gpu_support()

        print('>> fitting neural net...\n')
        kfold_output = cross_validate(
            pipeline, 
            x, 
            y, 
            cv = kfold_spooler, 
            return_train_score = True,
            return_estimator = True, 
            verbose = 0
        )

        print_cross_validation_scores(kfold_output)

        for kfold, pipeline in enumerate(kfold_output['estimator']):
            save_pipeline(
                tf_dir / f'net_{kfold:02d}.keras', 
                sk_dir / f'pipeline_{kfold:02d}.pickle',
                pipeline
            )

    else:

        check_gpu_support()

        print('>> fitting neural net...\n')
        pipeline.fit(x, y)

        save_pipeline(
            tf_dir / f'net.keras', 
            sk_dir / f'pipeline.pickle',
            pipeline
        )
    
    return 0

def predict(
    model_dir: str,
    x_path: str,
    conv_params: dict = {},
    **kwargs
):
    """
    PREDICT. A preprocessing pipeline and neural network are restored from a
    model.[?] directory created by the LEARN routine. The .xyz (X) data are
    loaded and transformed (via the preprocessing pipeline); the neural
    network is used to predict the corresponding XANES spectral (Y) data.
    Convolution (see xanesnet/convolute.py) of the XANES spectral data is
    possible if {conv_params} are provided.
    This function creates a predict.[?] directory in the current workspace:

    > predict.[?]
      ~ n [?].txt (predicted XANES spectral data)
      ~ n [?]_conv.txt (predicted XANES spectral data post-convolution)

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

    predict_dir = Path(f'./predict.{int(time.time())}')
    
    predict_dir.mkdir()

    model_dir = Path(model_dir)
    obj_dir = model_dir / 'objects'
    np_dir = obj_dir / 'np'
    tf_dir = obj_dir / 'tf'
    sk_dir = obj_dir / 'sk'

    with open(sk_dir / 'descriptor.pickle', 'rb') as f:
        descriptor = pickle.load(f)

    x_path = Path(x_path)  

    ids = load_data_ids(x_path)

    x_spooler = (x_path / (id_ + '.xyz') for id_ in ids)
    print('>> loading X data...')
    x = [descriptor.transform(load_xyz(f)) for f in tqdm.tqdm(x_spooler)]
    print()
   
    x = np.array(x, dtype = 'float32')

    with open(np_dir / 'e.npy', 'rb') as f:
        e = np.load(f)

    pipeline = load_pipeline(
        tf_dir / 'net.keras',
        sk_dir / 'pipeline.pickle'
    )

    print('>> predicting Y data with neural net...')
    y_predicts = pipeline.predict(x)
    print()
    
    print('>> saving Y data predictions...')
    for id_, y_predict in tqdm.tqdm(zip(ids, y_predicts)):
        save_xanes(predict_dir / f'{id_}.txt', e, y_predict)
    print()

    if conv_params:
        
        convoluter = ArctanConvoluter(**conv_params)

        print('>> convoluting and saving Y data predictions...')
        for id_, y_predict in tqdm.tqdm(zip(ids, y_predicts)):
            y_predict_conv = convoluter.convolute(e, y_predict)
            save_xanes(predict_dir / f'{id_}_conv.txt', e, y_predict_conv)
        print()
        
    return 0