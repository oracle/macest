# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
"""Module for regression evaluation metrics."""

import logging
import numpy as np
from macest.regression.models import ModelWithPredictionInterval

from typing import Union

log = logging.getLogger()


def predictions_in_range(
        test_true_val: np.ndarray,
        x_star: np.ndarray,
        interval_model: ModelWithPredictionInterval,
        conf_level: Union[np.ndarray, int, float] = 90,
        verbose: bool = False,
) -> float:
    """
    Test the predicted intervals against the known test values to estimate the interval calibration.

    :param test_true_val: The known values to compare with the predicted distributions
    :param x_star: The variables for which we would like to use to predict a distribution
    :param interval_model: A model which makes predictions with a standard deviation
    :param conf_level:   The interval contained within the upper and lower bounds (default 90)
    :param verbose: boolean switch to print results or not

    :return: The fraction of points within a specified confidence interval
    """
    lower, upper = interval_model.predict_interval(
        x_star, conf_level=np.array(conf_level)
    ).T
    lower = lower.flatten()
    upper = upper.flatten()
    in_range = (100 * len(np.where(np.logical_and(test_true_val >= lower, test_true_val <= upper))[0])
                / len(test_true_val))
    if verbose:
        log.info(f"percentage of values inside of interval {in_range:.0f}%")
    return in_range


def mean_prediction_interval_width(
        interval_model: ModelWithPredictionInterval,
        x_test: np.ndarray,
        conf_level: Union[np.ndarray, int, float] = 90,
) -> np.ndarray:
    """
    Calculate the mean prediction interval width (mean_prediction_interval_width), \
    measures the average size of all prediction intervals.

    :param interval_model: A  model which makes predictions with a standard deviation
    :param x_test: The variables for which we would like to use to predict a distribution
    :param conf_level: The interval contained within the upper and lower bounds (default 90)

    :return: Mean prediction interval width for x_test
    """
    intervals = interval_model.predict_interval(x_test, conf_level=conf_level).T
    return abs(intervals[0] - intervals[1]).mean()


def prediction_interval_coverage_probability(
        interval_model: ModelWithPredictionInterval,
        x_star: np.ndarray,
        y_true: np.ndarray,
        conf_level: Union[np.ndarray, int, float] = 90,
) -> float:
    """
     Calculate the prediction interval coverage probability (prediction_interval_coverage_probability) \
     representing the percentage of the time the prediction interval is correct.

    :param interval_model: A model which makes predictions with a standard deviation
    :param x_star: The variables for which we would like to use to predict a distribution
    :param y_true: The True target value
    :param conf_level:   The interval contained within the upper and lower bounds (default 90)

    :return: Prediction interval coverage probability for x_star
    """
    intervals = interval_model.predict_interval(x_star, conf_level=conf_level).T
    return 100 * len(np.where(np.logical_and(y_true >= intervals[0], y_true <= intervals[1]))[0]) / len(y_true)
