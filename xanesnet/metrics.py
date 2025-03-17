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

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

###############################################################################
################################## FUNCTIONS ##################################
###############################################################################

def get_metric(
    metric_type: str
) -> callable:
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
        'mae': mean_absolute_error
    }

    try:
        return metrics.get(metric_type)
    except KeyError:
        raise ValueError(
            f'\'{metric_type}\' is not a valid/supported metric'
        ) from None
