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

from xanesnet_core import check_gpu_support
from xanesnet_core import load_user_input_f
from xanesnet_core import load_data_ids
from xanesnet_core import xyz2x
from xanesnet_core import xas2y
from xanesnet_core import get_kf_idxs
from xanesnet_core import compile_mlp
from xanesnet_core import fit_net
from xanesnet_core import log2csv
from xanesnet_core import xas2csv
from xanesnet_core import metrics2csv
from xanesnet_descriptors import CoulombMatrixDescr
from xanesnet_descriptors import RadialDistCurveDescr
from xanesnet_convolute import ArctanConvoluter

###############################################################################
############################## RUNTIME ROUTINES ###############################
###############################################################################

def learn(inp_f):

    mdl_dir = Path('./model.{}'.format(int(time.time())))
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
        featuriser = CoulombMatrixDescr(inp['n_max'])
    elif inp['features'] == 'rdc':
        featuriser = RadialDistCurveDescr(inp['r_max'], inp['dr'], inp['alpha'])                                                          
    else:
        err_str = 'requested representation not implemented; got {}'
        raise ValueError(err_str.format(inp['features']))

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

        print('>> cycle no. {:.0f}/{:.0f}\n'.format(kf_n + 1, len(kf_idxs)))

        train_idxs, test_idxs = kf_idxs_pair

        x_train, y_train = zip(*[(x[i], y[i]) for i in train_idxs])
        x_test, y_test = zip(*[(x[i], y[i]) for i in test_idxs])

        scaler = StandardScaler()
        
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        with open(pkl_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)   

        kf_dir = tf_dir / 'k{}'.format(kf_n)
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

def predict(mdl_dir, xyz_dir, conv_inp_f = None):

    predict_dir = Path('./predict.{}'.format(int(time.time())))
    
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
        csv_f = predict_dir / '{}.csv'.format(id_)
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
            csv_f = predict_dir / '{}_conv.csv'.format(id_)
            xas2csv(e, convoluter.convolute(y), csv_f)
        print()