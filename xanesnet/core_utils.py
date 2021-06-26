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

from ase import Atoms
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def check_gpu_support():
    # checks if TensorFlow/Keras can identify GPUs/CUDA-compatible devices on 
    # the system; prints out a message and the identified devices 

    dev_type = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'

    str_ = '>> detecting GPU/nVidia CUDA acceleration support: {}\n'
    print(str_.format('supported' if dev_type == 'GPU' else 'unsupported'))

    print(f'>> listing available {dev_type} devices:')
    for device in tf.config.list_physical_devices(dev_type):
        print(f'>> {device[0]}') 

    print()

    return 0

def load_data_ids(*dirs: str) -> list:
    # returns a list of extensionless file names (used as data IDs) *if* the
    # list is common to all directories (*dirs; data sources) and not empty, 
    # otherwise raises a runtime error; prints out a message and the length of 
    # the list

    print('>> listing supplied data sources:')
    
    for i, d in enumerate(dirs):
        print(f'>> {i + 1}. {d}')

    print()

    ids = [sorted([f.stem for f in d.iterdir() if f.is_file()]) 
           for d in dirs]

    if ids.count(ids[0]) != len(ids) or len(ids[0]) == 0:
        raise RuntimeError('missing/mismatched files/IDs in data source(s)')
    else:
        ids = ids[0]

    print(f'>> loaded {len(ids)} IDs from the supplied data source(s)')
    
    print()

    return ids

def load_csv_input_f(csv_inp_f: str) -> list:
    # returns a list of (columnwise) lists from a .csv file (csv_inp_f); used
    # to load G2/G4 parameter files for setting up xanesnet.descriptors.WACSF
    # objects

    with open(csv_inp_f, 'r') as f:
        ls = [l.strip().split(',') for l in f if not l.startswith('#')]
    csv_cols = [list(col) for col in zip(*ls)]

    return csv_cols

def xyz2x(xyz_f: str, descriptor) -> np.ndarray:
    # returns the np.ndarray feature vector for an .xyz file; used to load X
    # data from a data source directory

    with open(xyz_f) as f:
        xyz_f_l = [l.strip().split() for l in f]

    z = np.array([l[0] for l in xyz_f_l[2:]], dtype = 'str')
    xyz = np.array([l[1:] for l in xyz_f_l[2:]], dtype = 'float32')
    
    try:
        ase = Atoms(z, xyz)
    except KeyError:
        ase = Atoms(z.astype('uint8'), xyz)

    features = descriptor.describe(ase)

    return features

def xas2y(xas_f: str) -> (np.ndarray, np.ndarray):
    # returns the np.ndarray XANES spectral components (e = energy scale,
    # mu = spectral intensity) for a .txt FDMNES output file; used to load Y
    # data from a data source directory

    with open(xas_f) as f:
        xas_f_l = [l.strip().split() for l in f]

    e = np.array([l[0] for l in xas_f_l[2:]], dtype = 'float32')
    mu = np.array([l[1] for l in xas_f_l[2:]], dtype = 'float32')

    mu /= mu[-1]

    return e, mu

def get_kf_idxs(ids: list, n_splits: int, 
                n_repeats: int) -> (np.ndarray, np.ndarray):
    # returns np.ndarray training and testing/validation K-fold splits;
    # a wrapper for scipy.model_selection.RepeatedKFold
    
    kf_spooler = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats)
 
    try:
        kf_idxs = list(kf_spooler.split(ids))
    except ValueError:
        kf_idxs = [tuple([np.linspace(0, len(ids) - 1, len(ids) - 1, 
                          dtype = 'uint32')] * 2) for _ in range(n_repeats)]

    return kf_idxs

def compile_mlp(inp_dim: int, 
                out_dim: int, 
                n_hl: int, 
                ini_hl_dim: int, 
                hl_shrink: float, 
                activation: str, 
                dropout: float, 
                lr: float, 
                kernel_init: str, 
                bias_init: str, 
                loss: str) -> Sequential:
    # returns a tensorflow.keras.models.Sequential neural network set up 
    # according to specification via input arguments

    net = Sequential()

    net.add(Dense(ini_hl_dim, input_dim = inp_dim,
                  kernel_initializer = kernel_init,
                  kernel_regularizer = None,
                  bias_initializer = bias_init,
                  bias_regularizer = None))
    net.add(Activation(activation))
    net.add(Dropout(dropout))
    
    for i in range(n_hl - 1):
        hl_dim = (int(ini_hl_dim * (hl_shrink ** (i + 1))))
        net.add(Dense(hl_dim if (hl_dim > 1) else 1, 
                      kernel_initializer = kernel_init,
                      kernel_regularizer = None,
                      bias_initializer = bias_init,
                      bias_regularizer = None))
        net.add(Activation(activation))
        net.add(Dropout(dropout))
    
    net.add(Dense(out_dim))
    net.add(Activation('linear'))

    net.compile(loss = loss, optimizer = Adam(lr = lr))

    return net

def fit_net(net: Sequential, 
            train_data: tuple, 
            test_data: tuple, 
            epochs: int, 
            batch_size: int, 
            chk_dir: str, 
            log_dir: str) -> (Sequential):

    chk = ModelCheckpoint(str(chk_dir / 'chk'),            
                          monitor = 'val_loss', 
                          save_weights_only = 'true',
                          save_best_only = 'true',
                          mode = 'auto',
                          verbose = 0)

    csvlog = CSVLogger(str(log_dir / 'log.csv'))

    callbacks = [chk, csvlog]

    net.fit(*train_data,
             epochs = epochs, 
             batch_size = batch_size, 
             callbacks = callbacks,
             validation_data = test_data,
             shuffle = True, 
             verbose = 2)
    
    net.load_weights(str(chk_dir / 'chk'))

    print()

    return net

def xas2csv(e: np.ndarray, mu: np.ndarray, xas_f: str):
    # writes a np.ndarray XANES spectrum (e = energy scale; mu = spectral 
    # intensity) in .csv format to a file (xas_f)

    fmt = ['%.2f'] + ['%.8f']
    header = 'e,mu'

    with open(xas_f, 'w') as f:
        np.savetxt(f, np.c_[e, mu / mu[-1]], delimiter = ',',
                   fmt = fmt, header = header)

    return 0

def metrics2csv(out_dir: str, tf_dir: str):
    # writes summary statistics derived from K-fold data in the TensorFlow
    # directory (tf_dir) in .csv format to files (out_dir/epochs.csv + 
    # out_dir/best.csv)

    print(f'>> saving net performance stats @ {out_dir}')

    metrics = np.array([np.genfromtxt(d / 'log' / 'metrics.csv', 
                        delimiter = ',')[:,1:] for d in tf_dir.iterdir()])
    
    n_logs, n_epochs, _ = metrics.shape
    kfs = np.linspace(1, n_logs, n_logs, dtype = 'uint16')
    epochs = np.linspace(1, n_epochs, n_epochs, dtype = 'uint16')

    metrics_avg = np.average(metrics, axis = 0)
    metrics_std = np.std(metrics, axis = 0)
    best = np.min(metrics, axis = 1)

    fmt = ['%.0f'] + ['%.6f'] * 4
    header = 'epochs,train,valid,train_std,valid_std'

    with open(out_dir / 'epochs.csv', 'w') as f:
        np.savetxt(f, np.c_[epochs, metrics_avg, metrics_std], 
                   delimiter = ',', fmt = fmt, header = header)

    fmt = ['%.0f'] + ['%.6f'] * 2
    header = 'kfold,train_best,valid_best'

    with open(out_dir / 'best.csv', 'w') as f:
        np.savetxt(f, np.c_[kfs, best], 
                   delimiter = ',', fmt = fmt, header = header)

    print()

    return 0