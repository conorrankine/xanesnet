"""
XANESNET-REDUX
Copyright (C) 2025  Conor D. Rankine

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
from typing import Callable
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def get_metric(
    metric_type: str
) -> Callable:
    """
    Returns a metric/"scoring" function of the specified type; a metric
    function has the form f(`y_true`, `y_predicted`, *) -> `loss`.

    Args:
        metric_type (str): Metric type, e.g., 'mse' (mean-squared error); 'mae'
            (mean absolute error); etc.

    Raises:
        ValueError: If `metric_type` is not not a valid/supported metric.

    Returns:
        callable: Metric function.
    """
    
    metrics = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'rse': relative_spectral_error
    }

    try:
        return metrics.get(metric_type)
    except KeyError:
        raise ValueError(
            f'\'{metric_type}\' is not a valid/supported metric'
        ) from None

def relative_spectral_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 0.2,
    sample_weight: np.ndarray = None,
    multioutput = 'uniform_average'
) -> float:
    """
    Relative spectral error (RSE) metric/"scoring" function for evaluating
    X-ray absorption spectra.

    Args:
        y_true (np.ndarray): Array of target absorption intensities (of shape
            `n_samples`, `n_bins`).
        y_pred (np.ndarray): Array of predicted absorption intensities (of
            shape `n_samples`, `n_bins`).
        delta (float, optional): 'Distance' in energy space (eV) between
            discrete energy bins. Defaults to 0.2.
        sample_weight (np.ndarray, optional): Sample weights. Defaults to None.
        multioutput (str, optional): Defines aggregating of multiple output
            values; options are 'raw_values' (all sample errors are returned
            without aggregation) or 'uniform_average' (a uniform average error
            over all sample errors is returned). Defaults to 'uniform_average'.

    Returns:
        float: Relative spectral error (RSE).
    """
    
    # TODO: implement sample weighting using `sample_weight`
    output_errors = (
        np.sqrt(np.sum(np.square(y_true - y_pred), axis = 1) * delta)
    ) / (
        (np.sum(y_pred, axis = 1) * delta)
    )

    if multioutput == 'raw_values':
        return output_errors
    elif multioutput == 'uniform_average':
        multioutput = None
    
    rse = np.average(output_errors)

    return rse
