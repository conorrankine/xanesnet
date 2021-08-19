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

from ase import Atoms
from pathlib import Path
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def load_xyz(xyz_f: Path) -> Atoms:
    # loads an ase.atoms object from a .xyz file

    with open(xyz_f, 'r') as f:
        xyz_f_l = [l.strip().split() for l in f]

    # atomic numbers
    z = np.array([l[0] for l in xyz_f_l[2:]], dtype = 'str')
    # atomic coordinates in .xyz format
    xyz = np.array([l[1:] for l in xyz_f_l[2:]], dtype = 'float32')
    
    try:
        return Atoms(z, xyz)
    except KeyError:
        return Atoms(z.astype('uint8'), xyz)

def save_xyz(xyz_f: Path, atoms: Atoms):
    # saves an ase.atoms object in .xyz format

    with open(xyz_f, 'w') as f:
        f.write(f'{len(atoms)}\n\n')
        for atom in atoms:
            fmt = '{:<4}{:>12.8f}{:>12.8f}{:>12.8f}\n'
            f.write(fmt.format(atom.symbol, *atom.position))

    return 0

def load_xanes(xanes_f: Path) -> (np.ndarray, np.ndarray):
    # loads XANES spectral data from a .txt FDMNES output file

    with open(xanes_f, 'r') as f:
        xanes_f_l = [l.strip().split() for l in f]

    # absorption energies
    e = np.array([l[0] for l in xanes_f_l[2:]], dtype = 'float32')
    # absorption intensities
    m = np.array([l[1] for l in xanes_f_l[2:]], dtype = 'float32')

    # TODO: move absorption intensity scaling out of this function
    m /= m[-1]

    return e, m

def save_xanes(xanes_f: Path, e: np.ndarray, m: np.ndarray):
    # saves XANES spectral data in .txt FDMNES output format

    with open(xanes_f, 'w') as f:
        f.write(f'{"FDMNES":>10}\n{"energy":>10}{"<xanes>":>12}\n')
        for e_, m_ in zip(e, m):
            fmt = f'{e_:>10.2f}{m_:>15.7E}\n'
            f.write(fmt.format(e_, m_))

    return 0

def load_data_ids(*dirs: Path) -> list:
    # returns a list of extensionless file names (used as data IDs) *if* the
    # list is common to all directories (*dirs; data sources) and not empty,
    # otherwise raises a runtime error; prints out a message and the length
    # of the list

    print('>> listing supplied data sources:')
    
    for i, d in enumerate(dirs):
        print(f'>> {i + 1}. {d}')

    print()

    ids = [sorted([f.stem for f in d.iterdir() if (f.is_file()
        and not f.stem.startswith('.'))]) for d in dirs]

    if ids.count(ids[0]) != len(ids) or len(ids[0]) == 0:
        raise RuntimeError('missing/mismatched files/IDs in data source(s)')
    else:
        ids = ids[0]

    print(f'>> loaded {len(ids)} IDs from the supplied data source(s)')
    
    print()

    return ids

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