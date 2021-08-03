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

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def print_cross_validation_scores(scores: dict):
    # prints a summary table of the scores from k-fold cross validation;
    # summarises the elapsed time and train/test metric scores for each k-fold
    # with overall k-fold cross validation statistics (mean and std. dev.)
    # using the `scores` dictionary returned from `cross_validate`

    print()
    print('>> summarising scores from k-fold cross validation...')
    print()

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