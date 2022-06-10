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
############################### LIBRARY IMPORTS ###############################t
###############################################################################

import numpy as np
import pickle as pickle

from ase import Atoms
from pathlib import Path
from sklearn.pipeline import Pipeline
from typing import TextIO
from typing import BinaryIO
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model

from xanesnet.utils import str_to_numeric

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def load_xyz(xyz_f: TextIO) -> Atoms:
    # loads an Atoms object from a .xyz file

    xyz_f_l = xyz_f.readlines()

    # pop the number of atoms `n_ats`
    n_ats = int(xyz_f_l.pop(0))

    # pop the .xyz comment block
    comment_block = xyz_f_l.pop(0)

    # pop the .xyz coordinate block
    coord_block = [xyz_f_l.pop(0).split() for _ in range(n_ats)]
    # atomic symbols or atomic numbers
    ats = np.array([l[0] for l in coord_block], dtype = 'str')
    # atomic coordinates in .xyz format
    xyz = np.array([l[1:] for l in coord_block], dtype = 'float32')

    try:
        info = dict([[key, str_to_numeric(val)] for key, val in
            [pair.split(' = ') for pair in comment_block.split(' | ')]])
    except ValueError:
        info = dict()
    
    try:
        # return Atoms object, assuming `ats` contains atomic symbols
        return Atoms(ats, xyz, info = info)
    except KeyError:
        # return Atoms object, assuming `ats` contains atomic numbers
        return Atoms(ats.astype('uint8'), xyz, info = info)

def save_xyz(xyz_f: TextIO, atoms: Atoms):
    # saves an Atoms object in .xyz format

    # write the number of atoms in `atoms`
    xyz_f.write(f'{len(atoms)}\n')
    # write additional info ('key = val', '|'-delimited) from the `atoms.info`
    # dictionary to the .xyz comment block
    for i, (key, val) in enumerate(atoms.info.items()):
        if i < len(atoms.info) - 1:
            xyz_f.write(f'{key} = {val} | ')
        else:
            xyz_f.write(f'{key} = {val}')
    xyz_f.write('\n')
    # write atomic symbols and atomic coordinates in .xyz format
    for atom in atoms:
        fmt = '{:<4}{:>16.8f}{:>16.8f}{:>16.8f}\n'
        xyz_f.write(fmt.format(atom.symbol, *atom.position))

    return 0

def load_xanes(xanes_f: TextIO) -> tuple:
    # loads XANES spectral data from a .txt FDMNES output file

    xanes_f_l = xanes_f.readlines()

    # pop the FDMNES header block
    for _ in range(2):
        xanes_f_l.pop(0)

    # pop the XANES spectrum block
    xanes_block = [xanes_f_l.pop(0).split() for _ in range(len(xanes_f_l))]
    # absorption energies
    e = np.array([l[0] for l in xanes_block], dtype = 'float32')
    # absorption intensities
    m = np.array([l[1] for l in xanes_block], dtype = 'float32')

    return e, m

def save_xanes(xanes_f: TextIO, e: np.ndarray, m: np.ndarray):
    # saves XANES spectral data in .txt FDMNES output format

    xanes_f.write(f'{"FDMNES":>10}\n{"energy":>10}{"<xanes>":>12}\n')
    for e_, m_ in zip(e, m):
        fmt = f'{e_:>10.2f}{m_:>15.7E}\n'
        xanes_f.write(fmt.format(e_, m_))

    return 0

def load_pipeline(keras_f: Path, pipeline_f: Path) -> Pipeline:
    # loads an sklearn pipeline with a Keras Sequential model; the pipeline
    # is reconstructed to add the Keras elements that were removed when the
    # mixed sklearn/Keras pipeline was pickled

    # load pipeline from pipeline_f
    with open(pipeline_f, 'rb') as f:
        pipeline = pickle.load(f)
    
    # load Keras model from keras_f and add the Keras elements to the pipeline
    # (pipeline.named_steps['net'].model)
    pipeline.named_steps['net'].model = load_model(keras_f)

    return pipeline

def save_pipeline(keras_f: Path, pipeline_f: Path, pipeline: Pipeline):
    # saves an sklearn pipeline with a Keras Sequential model; the pipeline
    # has to be deconstructed first to remove the Keras elements since mixed
    # sklearn/Keras pipelines cannot be pickled in the usual way

    # save Keras model (pipeline.named_steps['net'].model) to keras_f
    save_model(pipeline.named_steps['net'].model, keras_f)

    # set Keras model (pipeline.named_steps['net'].model) and callbacks
    # (pipeline.named_steps['net'].sk_params['callbacks']) to None so the
    # pipeline can be pickled
    pipeline.named_steps['net'].model = None
    pipeline.named_steps['net'].sk_params['callbacks'] = None

    # save pipeline to pipeline_f
    with open(pipeline_f, 'wb') as f:
        pickle.dump(pipeline, f)

    return 0
