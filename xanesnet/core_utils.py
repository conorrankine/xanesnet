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
from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation

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

def load_csv_f(csv_inp_f: str) -> list:
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

def compile_mlp(
    inp_dim: int, 
    out_dim: int, 
    n_hl: int, 
    hl_ini_dim: int, 
    hl_shrink: float, 
    activation: str, 
    loss: str,
    lr: float,
    dropout: float,
    kernel_init: str,
    bias_init: str
) -> Sequential:
    # returns a tensorflow.keras.models.Sequential neural network set up 
    # according to specification via input arguments

    net = Sequential()
    
    ini_condition = {
        "kernel_initializer": kernel_init,
        "kernel_regularizer": None,
        "bias_initializer": bias_init,
        "bias_regularizer": None        
    }

    net.add(Dense(hl_ini_dim, input_dim = inp_dim, **ini_condition))
    net.add(Activation(activation))
    net.add(Dropout(dropout))
    
    for i in range(n_hl - 1):
        hl_dim = (int(hl_ini_dim * (hl_shrink ** (i + 1))))
        net.add(Dense(hl_dim if (hl_dim > 1) else 1, **ini_condition))
        net.add(Activation(activation))
        net.add(Dropout(dropout))
    
    net.add(Dense(out_dim))
    net.add(Activation('linear'))

    net.compile(loss = loss, optimizer = Adam(lr = lr))

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

def metrics2csv(out_dir: Path, tf_dir: Path):
    # writes summary statistics derived from K-fold data in the TensorFlow
    # directory (tf_dir) in .csv format to files (out_dir/epochs.csv + 
    # out_dir/best.csv)

    logs = [np.genfromtxt(d / 'log' / 'log.csv', delimiter = ',',
                          skip_header = 1)[:,1:] for d in tf_dir.iterdir()]
    
    if not all([len(logs[0]) == len(log) for log in logs]):
        len_max = max([len(log) for log in logs])
        logs = [np.pad(log, ((0, len_max - len(log)), (0, 0)), mode = 'edge')
                for log in logs]
        
    logs = np.array(logs, dtype = 'float32') 

    log_avg = np.average(logs, axis = 0)
    log_std = np.std(logs, axis = 0)
    best = np.min(logs, axis = 1)
    
    n_kfs, n_epochs, _ = logs.shape
    
    epochs = np.linspace(1, n_epochs, n_epochs, dtype = 'uint16')
    fmt = ['%.0f'] + ['%.16f'] * 4
    header = 'epochs,loss,val_loss,loss_stdev,val_loss_stdev'
    with open(out_dir / 'epochs.csv', 'w') as f:
        np.savetxt(f, np.c_[epochs, log_avg, log_std], 
                   delimiter = ',', fmt = fmt, header = header)

    kfs = np.linspace(1, n_kfs, n_kfs, dtype = 'uint16')
    fmt = ['%.0f'] + ['%.16f'] * 2
    header = 'kfold,loss_best,val_loss_best'
    with open(out_dir / 'best.csv', 'w') as f:
        np.savetxt(f, np.c_[kfs, best], 
                   delimiter = ',', fmt = fmt, header = header)

    return 0