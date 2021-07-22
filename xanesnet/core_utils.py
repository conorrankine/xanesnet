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
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

def load_data_ids(*dirs: Path) -> list:
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

def xyz2ase(xyz_f: Path) -> Atoms:
    # loads an .xyz (X) data file as an ase.atoms object

    with open(xyz_f) as f:
        xyz_f_l = [l.strip().split() for l in f]

    z = np.array([l[0] for l in xyz_f_l[2:]], dtype = 'str')
    xyz = np.array([l[1:] for l in xyz_f_l[2:]], dtype = 'float32')
    
    try:
        return Atoms(z, xyz)
    except KeyError:
        return Atoms(z.astype('uint8'), xyz)

def txt2xas(txt_f: Path) -> (np.ndarray, np.ndarray):
    # loads a .txt FDMNES output (Y) data file as an np.ndarray object

    with open(txt_f) as f:
        txt_f_l = [l.strip().split() for l in f]

    e = np.array([l[0] for l in txt_f_l[2:]], dtype = 'float32')
    mu = np.array([l[1] for l in txt_f_l[2:]], dtype = 'float32')

    mu /= mu[-1]

    return e, mu

def get_kf_idxs(ids: list, n_splits: int, n_repeats: int) -> list:
    # returns two np.ndarrays containing training and testing/validation K-fold 
    # split indices; a wrapper for scipy.model_selection.RepeatedKFold (see
    # scikit-learn.org/stable/modules/generated/sklearn.model_selection \\
    # .KFold.html) that, if it fails to produce a split (e.g. if n_splits = 0), 
    # returns indices running over the whole list of IDs (ids) for both the
    # training and testing/validation K-fold splits in a format consistent with
    # the expected output from scipy.model_selection.RepeatedKFold
    # TODO: consider subclassing scipy.model_selection.RepeatedKFold to build 
    # in this functionality there; is it out of place here?
    
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
    n_hl: int = 2, 
    hl_ini_dim: int = 256, 
    hl_shrink: float = 0.5, 
    activation: str = 'relu', 
    loss: str = 'mse',
    lr: float = 0.001,
    dropout: float = 0.2,
    kernel_init: str = 'he_uniform',
    bias_init: str = 'zeros',
    **kwargs
) -> Sequential:
    # returns a tensorflow.keras.models.Sequential neural network with the deep
    # multilayer perceptron (MLP) model; the MLP has an input layer of 
    # [inp_dim] neurons and an output layer of [out_dim] neurons; there are 
    # [n_hl] hidden layers between the input and output layers, the first
    # hidden layer has [hl_ini_dim] neurons, and each successive hidden layer 
    # is reduced in size by a factor of [hl_shrink]

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

def compile_callbacks(**kwargs) -> list:
    # returns a list of tensorflow.keras.callbacks assembled from the **kwargs
    # passed to the function; expects dictionaries containing key/value pairs
    # to pass through to the appropriate tensorflow.keras.callbacks
    
    callbacks = []
    
    if 'callback_csvlogger' in kwargs:
        callbacks.append(
            CSVLogger(**kwargs['callback_csvlogger'])
        )
    if 'callback_modelcheckpoint' in kwargs:
        callbacks.append(
            ModelCheckpoint(**kwargs['callback_modelcheckpoint'])
        )
    if 'callback_earlystopping' in kwargs:
        callbacks.append(
            EarlyStopping(**kwargs['callback_earlystopping'])
        )
    if 'callback_reducelronplateau' in kwargs:
        callbacks.append(
            ReduceLROnPlateau(**kwargs['callback_reducelronplateau'])
        )
        
    return callbacks

def xas2csv(e: np.ndarray, mu: np.ndarray, xas_f: Path):
    # writes a XANES spectrum (e = energy scale; mu = spectral intensity) 
    # in .csv format to a file (xas_f)

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