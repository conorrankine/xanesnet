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
import tensorflow as tf
import pickle as pickle
import random as random
import tqdm as tqdm
import time as time

from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from xanesnet.core_utils import check_gpu_support
from xanesnet.core_utils import load_data_ids
from xanesnet.core_utils import load_csv_input_f
from xanesnet.core_utils import xyz2x
from xanesnet.core_utils import xas2y
from xanesnet.core_utils import get_kf_idxs
from xanesnet.core_utils import compile_mlp
from xanesnet.core_utils import fit_net
from xanesnet.core_utils import xas2csv
from xanesnet.core_utils import metrics2csv
from xanesnet.descriptors import CoulombMatrix
from xanesnet.descriptors import RadDistCurve
from xanesnet.descriptors import WACSF
from xanesnet.convolute import ArctanConvoluter

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def learn(x_dir: str,
          y_dir: str,
          features: str,
          feature_vars: dict,
          max_samples: int = 0,
          n_kf_splits: int = 0,
          n_kf_cycles: int = 1,
          n_hl: int = 1,
          ini_hl_dim: int = 100,
          hl_shrink: float = 1.0,
          activation: str = 'relu',
          loss: str = 'mse',
          lr: float = 0.001,
          dropout: float = 0.0,
          kernel_init: str = 'he_uniform',
          bias_init: str = 'zeros',
          epochs: int = 100,
          batch_size: int = 10,
          **kwargs):
    """
    LEARN. The .xyz (X) and XANES spectral (Y) data are loaded, featurised, 
    shuffled, scaled, and split into training and testing/validation k-folds; 
    a neural network is set up and fit to these data to find an Y <- X mapping.
    The runtime routine creates a model.[?] directory in the current workspace;
    this directory is organised hierarchically: 
    
    > model.[?]
      ~ model.hdf5 (an optimised TensorFlow/Keras model in .hdf5 format)
      > out (contains useful summary statistics in .csv format)
      > pkl (contains retained serialised objects in .pkl format)
      > tf (contains retained TensorFlow/Keras intermediate files)

    Args:
        x_dir (str): A path to the .xyz (X) data; expects a directory
            containing .xyz files 
        y_dir (str): A path to the XANES spectral (Y) data; expects a directory
            containing .txt FDMNES output files
        features (str): The type of featurisation to use, e.g. Coulomb matrices
            ('cmat'), radial distribution curves ('rdc'), etc.
        feature_vars (dict): The variable definitions required for the type of 
            featurisation as a dictionary of keywords
        max_samples (int, optional): The maximum number of (X|Y) data samples 
            to use; if 0, all available data samples are used. Defaults to 0.
        n_kf_splits (int, optional): The number of K-fold splits to use; if 0,
            K-fold CV is not used. Defaults to 0.
        n_kf_cycles (int, optional): The number of K-fold cycles to run; if 1,
            and n_kf_splits == 0, K-fold CV is not used. Defaults to 1.
        n_hl (int, optional): The number of dense hidden layers in the neural 
            network. Defaults to 0.
        ini_hl_dim (int, optional): The dimension (in neurons) of the initial
            hidden layer in the neural network. Defaults to 0.
        hl_shrink (float, optional): The hidden-layer-to-layer shrinkage factor
            for the neural network. Defaults to 0.0.
        activation (str, optional): The activation for the neural network (see 
            X). Defaults to 'relu'.
        loss (str, optional): The loss function for the neural network (see
            X). Defaults to 'mse'.
        lr (float, optional): The learning rate for the neural network. 
            Defaults to 0.001.
        dropout (float, optional): The dropout for the neural network (see 
            X). Defaults to 0.0.
        kernel_init (str, optional): The protocol for kernel initialisation in
            the neural network. Defaults to 'he_uniform'.
        bias_init (str, optional): The protocol for bias initialisation in
            the neural network. Defaults to 'zeros'.
        epochs (int, optional): The maximum number of epochs to train the
            neural network over. Defaults to 100.
        batch_size (int, optional): The minibatch size to use when calculating
            the gradient of the loss function. Defaults to 10.
    """

    mdl_dir = Path(f'./model.{int(time.time())}')
    out_dir = mdl_dir / 'out'
    pkl_dir = mdl_dir / 'pkl'
    tf_dir = mdl_dir / 'tf'

    for d in [mdl_dir, out_dir, pkl_dir, tf_dir]:
        d.mkdir()

    check_gpu_support()

    x_dir = Path(x_dir)
    y_dir = Path(y_dir)

    ids = load_data_ids(x_dir, y_dir)

    random.shuffle(ids)

    if max_samples:
        ids = ids[:max_samples]

    if features == 'cmat':
        featuriser = CoulombMatrix(**feature_vars)
    elif features == 'rdc':
        featuriser = RadDistCurve(**feature_vars)
    elif features == 'wacsf':
        featuriser = WACSF(**feature_vars)
    else:
        raise ValueError((f'\'{features}\' is not a recognised kind of '
                          'featurisation; check docs & examples for options'))

    with open(pkl_dir / 'featuriser.pkl', 'wb') as f:
        pickle.dump(featuriser, f)

    x_spooler = (x_dir / (id_ + '.xyz') for id_ in ids)
    print('>> spooling files to the xyz2x function...')
    x = [xyz2x(f, featuriser) for f in tqdm.tqdm(x_spooler)]
    print()

    y_spooler = (y_dir / (id_ + '.txt') for id_ in ids)
    print('>> spooling files to the xas2y function...')
    e, y = zip(*[xas2y(f) for f in tqdm.tqdm(y_spooler)])
    print()

    with open(pkl_dir / 'e_scale.pkl', 'wb') as f:
        pickle.dump(e[0], f)

    kf_idxs = get_kf_idxs(ids, n_kf_splits, n_kf_cycles)

    for kf_n, kf_idxs_pair in enumerate(kf_idxs):

        print(f'>> cycle no. {(kf_n + 1):.0f}/{(len(kf_idxs)):.0f}\n')

        train_idxs, test_idxs = kf_idxs_pair

        x_train, y_train = zip(*[(x[i], y[i]) for i in train_idxs])
        x_test, y_test = zip(*[(x[i], y[i]) for i in test_idxs])

        scaler = StandardScaler()
        
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        with open(pkl_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)   

        kf_dir = tf_dir / f'k{kf_n}'
        log_dir = kf_dir / 'log'
        chk_dir = kf_dir / 'chk'

        for d in [kf_dir, log_dir, chk_dir]:
            d.mkdir()

        net = compile_mlp(inp_dim = x_train[0].size, 
                          out_dim = y_train[0].size, 
                          n_hl = n_hl,
                          ini_hl_dim = ini_hl_dim,
                          hl_shrink = hl_shrink,
                          activation = activation,
                          dropout = dropout, 
                          lr = lr,
                          kernel_init = kernel_init,
                          bias_init = bias_init,
                          loss = loss)

        net = fit_net(net = net, 
                      train_data = (np.array(x_train, dtype = 'float64'), 
                                    np.array(y_train, dtype = 'float64')), 
                      test_data = (np.array(x_test, dtype = 'float64'), 
                                   np.array(y_test, dtype = 'float64')), 
                      epochs = epochs,
                      batch_size = batch_size,
                      chk_dir = chk_dir,
                      log_dir = log_dir)

        net.save(mdl_dir / 'net.hdf5')    

    metrics2csv(out_dir, tf_dir)
    
    return 0

def predict(mdl_dir: str, 
            xyz_dir: str, 
            conv_vars: dict = {},
            **kwargs):
    """
    PREDICT. The neural network is restored from the model.[?] directory created
    by launching the LEARN routine. The .xyz data for prediction are loaded, 
    featurised, and scaled consistently with the run that created the model.[?]
    directory. The neural network is used to predict the corresponding XANES
    spectra. Optionally, the predicted XANES spectra can be convoluted with an 
    energy-dependent arctan function (see xanesnet/convolute.py). The runtime
    routine creates a predict.[?] directory in the current workspace; this 
    directory contains the predicted XANES spectra.

    Args:
        mdl_dir (str): The path to a model.[?] directory created by launching
            the LEARN routine.
        xyz_dir (str): The path to a directory containing .xyz data; the neural
            network is used to predict the corresponding XANES spectra.        
        conv_inp_f (dict, optional): The variable definitions for arctan 
            convolution as a dictionary, i.e. the variables necessary to set up
            an ArctanConvoluter object (see xanesnet/convolute.py).
    """

    predict_dir = Path(f'./predict.{int(time.time())}')
    
    predict_dir.mkdir()

    check_gpu_support()

    mdl_dir = Path(mdl_dir)
    x_dir = Path(xyz_dir)  

    ids = load_data_ids(x_dir)

    with open(mdl_dir / 'pkl' / 'featuriser.pkl', 'rb') as f:
        featuriser = pickle.load(f)

    x_spooler = (x_dir / (id_ + '.xyz') for id_ in ids)
    print('>> spooling files to the xyz2x function...')
    x = [xyz2x(f, featuriser) for f in tqdm.tqdm(x_spooler)]
    print()
   
    with open(mdl_dir / 'pkl' / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    x = scaler.transform(x)

    net = tf.keras.models.load_model(mdl_dir / 'net.hdf5')

    y_predict = net.predict(np.array(x, dtype = 'float64'))

    with open(mdl_dir / 'pkl' / 'e_scale.pkl', 'rb') as f:
        e = pickle.load(f)

    print('>> spooling predictions to the xas2csv function...')
    for id_, y in tqdm.tqdm(zip(ids, y_predict)):
        csv_f = predict_dir / f'{id_}.csv'
        xas2csv(e, y, csv_f)
    print()

    if conv_vars:
        
        convoluter = ArctanConvoluter(e, **conv_vars)

        print('>> spooling conv. predictions to the xas2csv function...')
        for id_, y in tqdm.tqdm(zip(ids, y_predict)):
            csv_f = predict_dir / f'{id_}_conv.csv'
            xas2csv(e, convoluter.convolute(y), csv_f)
        print()
        
    return 0