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

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
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

def set_callbacks(**kwargs) -> list:
    # returns a list of tensorflow.keras.callbacks assembled from the **kwargs
    # passed to the function; expects dictionaries containing key/value pairs
    # to pass through to the appropriate tensorflow.keras.callbacks
    
    callbacks_ = {
        'csvlogger': CSVLogger,
        'earlystopping': EarlyStopping,
        'reducelronplateau': ReduceLROnPlateau
    }

    callbacks = []
    
    for callback_label, callback_ in callbacks_.items():
        if callback_label in kwargs:
            callbacks.append(callback_(**kwargs[callback_label]))

    return callbacks

def build_mlp(
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