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

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

###############################################################################
################################## CLASSES ####################################
###############################################################################

class CentreScaler(BaseEstimator, TransformerMixin):
    """
    Centres features using the 
    """

    def __init__(self, *, copy: bool = True):
        """
        Args:
            copy (bool, optional): Toggle between copy/in-place transformation.
                Defaults to True.
        """

        self.copy = copy

    def _reset(self):
        """
        Resets the internal data-dependent state of the scaler; __init__
        parameters are not touched.
        """

        if hasattr(self, "mean_"):
            for var in [self.mean_, self.sum_, self.n_samples_seen_]:
                del var   

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Computes the featurewise mean for subsequent centring.

        Args:
            X (np.ndarray): X data (2D array; n_samples, n_features) used to
                compute the mean.
            y (np.ndarray, optional): Ignored. 
                Defaults to None.

        Returns:
            self: Fitted CentreScaler object.
        """

        self._reset()

        return self.partial_fit(X, y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Computes the featurewise mean for subsequent centring online.

        Args:
            X (np.ndarray): X data (2D array; n_samples, n_features) used to
                compute the mean.
            y (np.ndarray, optional): Ignored. 
                Defaults to None.

        Returns:
            self: Fitted CentreScaler object.
        """

        first_pass = not hasattr(self, "n_samples_seen_")

        X = self._validate_data(
            X,
            reset = first_pass,
            accept_sparse = False,
            estimator = self,
            force_all_finite = 'allow-nan'
        )

        if first_pass:
            self.sum_ = np.nansum(X, axis = 0)
            self.n_samples_seen_ = X.shape[0]
        else:
            self.sum_ += np.nansum(X, axis = 0)
            self.n_samples_seen_ += X.shape[0]

        self.mean_ = self.sum_ / self.n_samples_seen_

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Centres X data.

        Args:
            X (np.ndarray): X data (2D array; n_samples, n_features) that
                should be centred featurewise.

        Returns:
            np.ndarray: Centred X data (2D array; n_samples, n_features).
        """

        X = self._validate_data(
            X,
            reset = False,
            accept_sparse = False,
            copy = self.copy,
            estimator = self,
            force_all_finite = 'allow-nan'
        )

        X -= self.mean_

        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverts centred X data to original, pre-centred state.

        Args:
            X (np.ndarray): X data (2D array; n_samples, n_features) that
                should be reverted to original, pre-centred state.

        Returns:
            np.ndarray: X data (2D array; n_samples, n_features) reverted
                to original, pre-centred state.
        """

        X = self._validate_data(
            X,
            reset = False,
            accept_sparse = False,
            copy = self.copy,
            estimator = self,
            force_all_finite = 'allow-nan'
        )

        X += self.mean_

        return X

class GroupMaxAbsScaler(BaseEstimator, TransformerMixin):
    """
    Scales a group, or 'slice', of features by the maximum absolute value found
    in that group of features; leaves all other features unscaled.
    
    After scaling, the maximum absolute value found in that group of features
    will be 1.0 (with the default group weighting), and so all values in that
    group of features will be defined in the range -1.0 -> 1.0. Using a
    different group weighting will scale the maximum absolute value so that
    the group of features is defined over a narrower (group weighting < 1.0)
    or wider (group weighting > 1.0) range. 
    
    The data are not otherwise centred or shifted. NaNs are treated as missing
    values, i.e. disregarded in `fit` and maintained in `transform`.
    """

    def __init__(
        self,
        *,
        group_idx: list = None,
        group_weight: float = 1.0,
        copy: bool = True
    ):
        """
        Args:
            group_idx (list, optional): The min/max indices defining the group,
                or 'slice', of features that should be scaled by the maximum
                absolute value found in that group of features. If None, all
                features are scaled by the global maximum absolute value.
                Defaults to None.
            group_weight (float, optional): The weight for the group, or
                'slice', of features. After scaling, the maximum absolute value
                found in that group of features with be 1.0 * group_weight.
                Defaults to 1.0.
            copy (bool, optional): Toggle between copy/in-place transformation.
                Defaults to True.
        """

        self.copy = copy

        if (isinstance(group_idx, (list, tuple)) and len(group_idx) == 2 and 
            all([isinstance(i, int) for i in group_idx])):
                self.group_idx = group_idx
        else:
            raise ValueError(f'expected group_idx: list/tuple containing two '
                'indices; got {group_idx}')

        if isinstance(group_weight, (int, float)):
            self.group_weight = float(group_weight)
        else:
            raise ValueError(f'expected group_weight: int/float; ',
                'got {group_weight}')

    def _reset(self):
        """
        Resets the internal data-dependent state of the scaler; __init__
        parameters are not touched.
        """

        if hasattr(self, "max_abs_"):
            for var in [self.max_abs_, self.n_samples_seen_]:
                del var

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Computes the maximum absolute value found in a feature group for
        subsequent scaling.

        Args:
            X (np.ndarray): X data (2D array; n_samples, n_features) used to
                compute the maximum absolute value found in a feature group.
            y (np.ndarray, optional): Ignored. 
                Defaults to None.

        Returns:
            self: Fitted GroupMaxAbsScaler object.
        """

        self._reset()

        return self.partial_fit(X, y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Computes the maximum absolute value found in a feature group for
        subsequent scaling online.

        Args:
            X (np.ndarray): X data (2D array; n_samples, n_features) used to
                compute the maximum absolute value found in a feature group.
            y (np.ndarray, optional): Ignored. 
                Defaults to None.

        Returns:
            self: Fitted GroupMaxAbsScaler object.
        """

        first_pass = not hasattr(self, "n_samples_seen_")

        X = self._validate_data(
            X,
            reset = first_pass,
            accept_sparse = False,
            estimator = self,
            force_all_finite = 'allow-nan'
        )

        if not self.group_idx:
            max_abs_ = np.nanmax(np.absolute(X))
        else:
            idx_i, idx_f = self.group_idx
            max_abs_ = np.nanmax(np.absolute(X[:,idx_i:idx_f]))

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else: 
            max_abs_ = np.max(self.max_abs_, max_abs_)
            self.n_samples_seen_ += X.shape[0]

        self.max_abs_ = max_abs_

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scales X data.

        Args:
            X (np.ndarray): X data (2D array; n_samples, n_features) that
                should be scaled.

        Returns:
            np.ndarray: Scaled X data (2D array; n_samples, n_features).
        """

        X = self._validate_data(
            X,
            reset = False,
            accept_sparse = False,
            copy = self.copy,
            estimator = self,
            force_all_finite = 'allow-nan'
        )

        if not self.group_idx:
            X /= (self.max_abs_ / self.group_weight)
        else:
            idx_i, idx_f = self.group_idx
            X[:,idx_i:idx_f] /= (self.max_abs_ / self.group_weight)

        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverts scaled X data to original, pre-scaled state.

        Args:
            X (np.ndarray): X data (2D array; n_samples, n_features) that
                should be reverted to original, pre-scaled state.

        Returns:
            np.ndarray: X data (2D array; n_samples, n_features) reverted
                to original, pre-scaled state.
        """

        X = self._validate_data(
            X,
            reset = False,
            accept_sparse = False,
            copy = self.copy,
            estimator = self,
            force_all_finite = 'allow-nan'
        )

        if not self.group_idx:
            X *= (self.max_abs_ / self.group_weight)
        else:
            idx_i, idx_f = self.group_idx
            X[:,idx_i:idx_f] *= (self.max_abs_ / self.group_weight)

        return X