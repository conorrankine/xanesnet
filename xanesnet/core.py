###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np
import tensorflow as tf
import ast as ast

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from ase import Atoms

###############################################################################
############################### CORE FUNCTIONS ################################
###############################################################################

def check_gpu_support():

    dev_type = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'

    str_ = '>> detecting GPU/nVidia CUDA acceleration support: {}\n'
    print(str_.format('supported' if dev_type == 'GPU' else 'unsupported'))

    print('>> listing available {} devices:'.format(dev_type))
    for device in tf.config.list_physical_devices(dev_type):
        print('>> {}'.format(device[0])) 

    print()

    return 0

def load_user_input_f(inp_f):

    print('>> loading user input @ {}\n'.format(inp_f))
    
    inp = {}
    
    with open(inp_f) as f:
        ls = [l for l in f if l.strip() and not l.startswith('#')]

    for l in ls:
        (var, val) = l.split('=')
        print('>> {} :: {}'.format(var.strip(), val.strip()))
        try:
            inp[var.strip()] = ast.literal_eval(val.strip())
        except ValueError:
            inp[var.strip()] = val.strip()

    print()

    return inp

def load_data_ids(*dirs):

    print('>> listing supplied data sources:')
    
    for i, d in enumerate(dirs):
        print('>> {}. {}'.format(i + 1, d))
    
    print()

    ids = [sorted([f.stem for f in d.iterdir() if f.is_file()]) 
           for d in dirs]

    if ids.count(ids[0]) != len(ids) or len(ids[0]) == 0:
        raise RuntimeError('missing/mismatched files/IDs in data source(s)')
    else:
        ids = ids[0]

    print('>> loaded {} IDs in the supplied data source(s)'.format(len(ids)))
    
    print()

    return ids

def xyz2x(xyz_f, descriptor):

    with open(xyz_f) as f:
        xyz_f_l = [l.strip().split() for l in f]

    z = np.array([l[0] for l in xyz_f_l[2:]], dtype = 'str')
    xyz = np.array([l[1:] for l in xyz_f_l[2:]], dtype = 'float64')
    
    try:
        ase = Atoms(z, xyz)
    except KeyError:
        ase = Atoms(z.astype('float64'), xyz)

    features = descriptor.describe(ase)

    return features

def xas2y(xas_f):

    with open(xas_f) as f:
        xas_f_l = [l.strip().split() for l in f]

    e = np.array([l[0] for l in xas_f_l[2:]], dtype = 'float64')
    mu = np.array([l[1] for l in xas_f_l[2:]], dtype = 'float64')

    mu /= mu[-1]

    return e, mu

def get_kf_idxs(ids, n_splits, n_repeats):

    kf_spooler = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats)
 
    try:
        kf_idxs = list(kf_spooler.split(ids))
    except ValueError:
        kf_idxs = [tuple([np.linspace(0, len(ids) - 1, len(ids) - 1, 
                          dtype = 'uint32')] * 2) for _ in range(n_repeats)]

    return kf_idxs

def compile_mlp(inp_dim, out_dim, n_hl, ini_hl_dim, hl_shrink, activation,
                dropout, lr, kernel_init, bias_init, loss):

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

def fit_net(net, train_data, test_data, epochs, batch_size,
            chk_dir, log_dir):

    chk = ModelCheckpoint(str(chk_dir / 'chk'),            
                          monitor = 'val_loss', 
                          save_weights_only = 'true',
                          save_best_only = 'true',
                          mode = 'auto',
                          verbose = 0)

    callbacks = [chk]

    log = net.fit(*train_data,
                  epochs = epochs, 
                  batch_size = batch_size, 
                  callbacks = callbacks,
                  validation_data = test_data,
                  shuffle = True, 
                  verbose = 2)
    
    net.load_weights(str(chk_dir / 'chk'))

    print()

    return net, log

def xas2csv(e, mu, xas_f):

    fmt = ['%.2f'] + ['%.8f']
    header = 'e,mu'

    with open(xas_f, 'w') as f:
        np.savetxt(f, np.c_[e, mu / mu[-1]], delimiter = ',',
                   fmt = fmt, header = header)

    return 0

def log2csv(log, log_dir):

    print('>> saving net perf. logs @ {}'.format(log_dir))

    _, logs = zip(*log.history.items())
    logs = np.array(logs, dtype = 'float64').T
    
    n_epochs, _ = logs.shape   
    epochs = np.linspace(1, n_epochs, n_epochs, dtype = 'uint16')

    fmt = ['%.0f'] + ['%.6f'] * 2
    header = 'epochs,train,valid'

    with open(log_dir / 'metrics.csv', 'w') as f:
        np.savetxt(f, np.c_[epochs, logs], delimiter = ',', 
                   fmt = fmt, header = header)

    print()

    return 0

def metrics2csv(out_dir, tf_dir):

    print('>> saving net perf. stats @ {}'.format(out_dir))

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