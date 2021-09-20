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

from pathlib import Path

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def load_file_stems(*dirs: Path, verbose: bool = True) -> np.ndarray:
    # returns a single sorted np.ndarray (dtype: string) of file stems *if* 
    # the sorted lists of file stems are the same for all supplied directories
    # (`*dirs`) *and* the supplied directories are not empty, otherwise raises
    # a value error; prints out the supplied directory paths and the number of
    # file stems in the np.ndarray if `verbose` is True

    if verbose:
        print('>> loading file stems from the supplied source(s):')   
        for i, d in enumerate(dirs): print(f'  >> {i + 1}. {d}')

    file_stems = [sorted([f.stem for f in d.iterdir() if (f.is_file()
        and not f.stem.startswith('.'))]) for d in dirs]

    if file_stems.count(file_stems[0]) != len(file_stems):
        raise ValueError('supplied source(s) don\'t have matching sorted '
            'lists of file stems; does every sample in one source have a '
            'corresponding sample in another with the same file stem?')
    elif len(file_stems[0]) == 0:
        raise ValueError('supplied source(s) are empty')
    
    file_stems = np.array(file_stems[0], dtype = 'str')

    if verbose:
        print(f'>> loaded {len(file_stems)} file stems\n')
    
    return file_stems

def print_nested_dict(dict_: dict, nested_level: int = 0):
    # prints the key:value pairs in a dictionary (`dict`) in the format
    # '>> key :: value'; iterates recursively through any subdictionaries,
    # indenting with two white spaces for each sublevel (`nested level`)

    for key, val in dict_.items():
        if not isinstance(val, dict):
            print('  ' * nested_level + f'>> {key} :: {val}')
        else:
            print('  ' * nested_level + f'>> {key}')
            print_nested_dict(val, nested_level = nested_level + 1)

    return 0

def print_cross_validation_scores(scores: dict):
    # prints a summary table of the scores from k-fold cross validation;
    # summarises the elapsed time and train/test metric scores for each k-fold
    # with overall k-fold cross validation statistics (mean and std. dev.)
    # using the `scores` dictionary returned from `cross_validate`

    print('')
    print('>> summarising scores from k-fold cross validation...')
    print('')

    print('*' * 36)
    
    fmt = '{:<10s}{:>6s}{:>10s}{:>10s}'
    print(fmt.format('k-fold', 'time', 'train', 'test'))
    
    print('*' * 36)

    fmt = '{:<10.0f}{:>5.1f}s{:>10.4f}{:>10.4f}'
    for kf, (t, train, test) in enumerate(zip(
        scores['fit_time'], scores['train_score'], scores['test_score'])):
        print(fmt.format(kf, t, np.absolute(train), np.absolute(test)))

    print('*' * 36)

    fmt = '{:<10s}{:>5.1f}s{:>10.4f}{:>10.4f}'
    means_ = (np.mean(np.absolute(scores[score])) 
        for score in ('fit_time', 'train_score', 'test_score'))
    print(fmt.format('mean', *means_))
    stdevs_ = (np.std(np.absolute(scores[score])) 
        for score in ('fit_time', 'train_score', 'test_score'))
    print(fmt.format('std. dev.', *stdevs_))

    print('*' * 36)

    return 0