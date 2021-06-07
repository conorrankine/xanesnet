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
from xanesnet.core_utils import load_user_input_f
from xanesnet.core_utils import load_data_ids
from xanesnet.core_utils import load_csv_input_f
from xanesnet.core_utils import xyz2x
from xanesnet.core_utils import xas2y
from xanesnet.core_utils import get_kf_idxs
from xanesnet.core_utils import compile_mlp
from xanesnet.core_utils import fit_net
from xanesnet.core_utils import log2csv
from xanesnet.core_utils import xas2csv
from xanesnet.core_utils import metrics2csv
from xanesnet.descriptors import CoulombMatrix
from xanesnet.descriptors import RadDistCurve
from xanesnet.descriptors import WACSF
from xanesnet.convolute import ArctanConvoluter

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def learn(inp_f: str):
    """
    The main runtime routine for 'learn' mode. The data are loaded, transformed
    into descriptors, shuffled, scaled, and split into training and validation 
    k-folds - all as specified in the input file (see examples & docs). A  
    neural network is instantiated (also as specified in the input file) and 
    fit to the data. 
    
    This runtime routine creates a model.xxxxxxx directory in
    the current workspace; this directory is organised hierarchically: 
    
    > model.xxxxxxxx
      ~ model.hdf5 (the optimised TensorFlow/Keras model in .hdf5 format)
      > out (contains useful summary statistics in .csv format)
      > pkl (contains retained serialised objects in .pkl format)
      > tf (contains retained TensorFlow/Keras intermediate files)
      
    The model.xxxxxxxx directory is required to launch the main runtime routine
    for 'predict' mode as many of the contents have to be loaded.

    Args:
        inp_f (str): The path to a .txt input file with variable definitions
                     (see examples & docs).
    """

    mdl_dir = Path(f'./model.{int(time.time())}')
    out_dir = mdl_dir / 'out'
    pkl_dir = mdl_dir / 'pkl'
    tf_dir = mdl_dir / 'tf'

    for d in [mdl_dir, out_dir, pkl_dir, tf_dir]:
        d.mkdir()

    check_gpu_support()

    inp = load_user_input_f(inp_f)

    x_dir = Path(inp['x_dir'])
    y_dir = Path(inp['y_dir'])

    ids = load_data_ids(x_dir, y_dir)

    random.shuffle(ids)

    if inp['max_samples']:
        ids = ids[:inp['max_samples']]

    if inp['features'] == 'cmat':
        featuriser = CoulombMatrix(inp['n_max'])
    elif inp['features'] == 'rdc':
        featuriser = RadDistCurve(inp['r_max'], inp['gridsize'], inp['alpha'])
    elif inp['features'] == 'wacsf':
        g2_vars = (load_csv_input_f(inp['g2_var_f']) 
                   if 'g2_var_f' in inp else None)
        g4_vars = (load_csv_input_f(inp['g4_var_f']) 
                   if 'g4_var_f' in inp else None)
        featuriser = WACSF(inp['r_max'], g2_vars = g2_vars, g4_vars = g4_vars)                                                         
    else:
        raise ValueError((f'{inp["features"]} is not implemented as a ',
                          'featurisation type; use \'cmat\', \'rdc\', or ',
                          '\'wacsf\''))

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

    kf_idxs = get_kf_idxs(ids, inp['n_splits'], inp['n_repeats'])

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
                          n_hl = inp['n_hl'],
                          ini_hl_dim = inp['ini_hl_dim'],
                          hl_shrink = inp['hl_shrink'],
                          activation = inp['activation'],
                          dropout = inp['dropout'], 
                          lr = inp['lr'],
                          kernel_init = inp['kernel_init'],
                          bias_init = inp['bias_init'],
                          loss = inp['loss'])

        net, log = fit_net(net = net, 
                           train_data = (np.array(x_train, dtype = 'float64'), 
                                         np.array(y_train, dtype = 'float64')), 
                           test_data = (np.array(x_test, dtype = 'float64'), 
                                        np.array(y_test, dtype = 'float64')), 
                           epochs = inp['epochs'],
                           batch_size = inp['batch_size'],
                           chk_dir = chk_dir,
                           log_dir = log_dir)

        net.save(mdl_dir / 'net.hdf5')

        log2csv(log, log_dir)     

    metrics2csv(out_dir, tf_dir)
    
    return 0

def predict(mdl_dir: str, xyz_dir: str, conv_inp_f = None):
    """
    The main runtime routine for 'predict' mode. A neural network is restored 
    from a model.xxxxxxxx directory created by launching the runtime routine 
    for 'learn' mode. The data for prediction are loaded, transformed into 
    descriptors, and scaled - all consistently with the run that created
    the model.xxxxxxxx directory. The neural network is then used to predict 
    the XANES spectra. Optionally, the predicted XANES spectra can be 
    convoluted with an energy-dependent arctan function (see convolute.py). 
    
    This runtime routine creates a predict.xxxxxxx directory in the current 
    workspace; this directory contains the predictions from the neural network.

    Args:
        mdl_dir (str): The path to a model.xxxxxxxx directory generated 
                       using the 'learn' runtime routine.
        xyz_dir (str): The path to a directory with .xyzs; XANES spectra will
                       be predicted for each .xyz.
        conv_inp_f (str, optional): The path to a .txt input file with variable
                                    definitions for arctan convolution (see
                                    examples & docs). Defaults to None.
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

    if conv_inp_f:

        conv_inp = load_user_input_f(conv_inp_f)
        
        convoluter = ArctanConvoluter(e, e_edge = conv_inp['e_edge'],
                                         e_l = conv_inp['e_l'],
                                         e_c = conv_inp['e_c'],
                                         e_f = conv_inp['e_f'],
                                         g_hole = conv_inp['g_hole'],
                                         g_max = conv_inp['g_max'])

        print('>> spooling conv. predictions to the xas2csv function...')
        for id_, y in tqdm.tqdm(zip(ids, y_predict)):
            csv_f = predict_dir / f'{id_}_conv.csv'
            xas2csv(e, convoluter.convolute(y), csv_f)
        print()
        
    return 0