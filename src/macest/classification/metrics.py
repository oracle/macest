# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
"""Module for evaluation metrics."""

from macest.classification.utils import histogram_max_conf_pred
from typing import Callable, Sequence, Union

import numpy as np

CalibFuncType = Callable[
        [
            Union[Sequence[int], np.ndarray],
            Union[Sequence[int], np.ndarray],
            np.ndarray,
            int,
            int,
        ],
        np.ndarray,
    ]


def average_calibration_error(
    model_preds: Union[Sequence[int], np.ndarray],
    true: Union[Sequence[int], np.ndarray],
    conf_preds: np.ndarray,
    n_bins: int = 10,
    min_bin_size: int = 1,
) -> float:
    """
    Calculate the Average Calibration Error.

    This histograms the predictions into uniformly spaced bins
    then compares the difference between the average confidence in those quantiles
    and the empirical accuracy of that bin

    :param model_preds: The point prediction from the sklearn-like model
    :param true: The true target variable
    :param conf_preds: The predicted confidence for the point prediction
    :param n_bins: The number of bins to use for binning the confidence scores
    :param min_bin_size: the minimum number of elements in a min before it is merged with the one to the right

    :return: The average calibration error for a set of confidence estimates

    """
    bin_mid, accuracy, count, conf, _ = histogram_max_conf_pred(
        true,
        model_preds,
        conf_preds,
        bin_method="uniform",
        n_bins=n_bins,
        min_bin_size=min_bin_size,
        check_conflicting_preds=False,
    )
    return abs(accuracy - conf).mean() * 100


def expected_calibration_error(
    model_preds: Union[Sequence[int], np.ndarray],
    true: Union[Sequence[int], np.ndarray],
    conf_preds: np.ndarray,
    n_bins: int = 10,
    min_bin_size: int = 1,
) -> float:
    """
    Calculate the Expected Calibration Error.

    This histograms the predictions into uniformly spaced
    bins then compares the difference between the average confidence in those quantiles and the
    empirical accuracy of that bin, these are then weighed by the count in each bin

    :param model_preds: The point prediction from the sklearn-like model
    :param true: The true target variable
    :param conf_preds: The predicted confidence for the point prediction
    :param n_bins: The number of bins to use for binning the confidence scores
    :param min_bin_size: the minimum number of elements in a min before it is merged with the one to the right

    :return: The expected calibration error for a set of confidence estimates
    """
    bin_mid, accuracy, count, conf, _ = histogram_max_conf_pred(
        true,
        model_preds,
        conf_preds,
        bin_method="uniform",
        n_bins=n_bins,
        min_bin_size=min_bin_size,
        check_conflicting_preds=False,
    )

    return (abs(accuracy - conf).dot(count)) / count.sum() * 100


def quantile_calibration_error(
    model_preds: Union[Sequence[int], np.ndarray],
    true: Union[Sequence[int], np.ndarray],
    conf_preds: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate the Quantile Calibration Error.

    This histograms the predictions into quantiles
    then compares find the difference between the average confidence in those quantiles and
    the empirical accuracy of that quantile

    :param model_preds: The point prediction from the sklearn-like model
    :param true: The true target variable
    :param conf_preds: The predicted confidence for the point prediction
    :param n_bins: The number of bins to use for binning the confidence scores

    :return: The quantile calibration error for a set of confidence estimates
    """
    bin_mid, accuracy, count, conf, _ = histogram_max_conf_pred(
        true,
        model_preds,
        conf_preds,
        bin_method="quantile",
        n_bins=n_bins,
        check_conflicting_preds=False,
    )
    return abs(accuracy - conf).mean() * 100


def class_expected_calibration_error(
    class_to_calculate: int,
    model_preds: Union[Sequence[int], np.ndarray],
    true: Union[Sequence[int], np.ndarray],
    conf_preds: np.ndarray,
    n_bins: int = 10,
    cls_of_prediction: bool = True,
    min_bin_size: int = 1,
) -> float:
    """
    Calculate the Class Expected Calibration Error.

    This computes the expected_calibration_error for a particular class.

    :param class_to_calculate: The class of predictions that we compute for
    :param model_preds: The point prediction from the sklearn-like model
    :param true: The true target variable
    :param conf_preds: The predicted confidence for the point prediction
    :param n_bins: The number of bins to use for binning the confidence scores
    :param cls_of_prediction: If True calculate the CCE for cases where the prediction was cls not not the true value
    :param min_bin_size: the minimum number of elements in a min before it is merged with the one to the right

    :return: The expected calibration error for a set of confidence estimates for a given class

    """
    if cls_of_prediction:
        idx_of_cls = model_preds == class_to_calculate
    else:
        idx_of_cls = true == class_to_calculate

    bin_mid, accuracy, count, conf, _ = histogram_max_conf_pred(
        true[idx_of_cls],
        model_preds[idx_of_cls],
        conf_preds[idx_of_cls],
        bin_method="uniform",
        n_bins=n_bins,
        min_bin_size=min_bin_size,
        check_conflicting_preds=False,
    )

    return (abs(accuracy - conf).dot(count)) / count.sum() * 100


def class_wise_expected_calibration_error(
    model_preds: Union[Sequence[int], np.ndarray],
    true: Union[Sequence[int], np.ndarray],
    conf_preds: np.ndarray,
    n_bins: int = 10,
    n_classes: int = 2,
    cls_of_prediction: bool = True,
    min_bin_size: int = 1,
) -> float:
    """
    Calculate the Class-wise Expected Calibration Error.

    This computes the expected_calibration_error for each class and takes
    the combined mean.

    :param model_preds: The point prediction from the sklearn-like model
    :param true: The true target variable
    :param conf_preds: The predicted confidence for the point prediction
    :param n_bins: The number of bins to use for binning the confidence scores
    :param n_classes: The number of possible classes in the problem
    :param cls_of_prediction: If True calculate the CCE for cases where the prediction was
                                class_to_calculate not not the true value
    :param min_bin_size: the minimum number of elements in a min before it is merged with the one to the right

    :return: The average expected calibration error for a set of confidence estimates across all classes
    """
    class_wise_exp_calib_err = np.zeros(n_classes)
    for class_to_calculate in range(n_classes):
        class_to_calculate = int(class_to_calculate)
        class_wise_exp_calib_err[class_to_calculate] = class_expected_calibration_error(
            class_to_calculate,
            model_preds,
            true,
            conf_preds,
            n_bins,
            cls_of_prediction,
            min_bin_size,
        )
    return class_wise_exp_calib_err.mean()


def simulate_from_calibrated_model(
    conf_preds: np.ndarray,
    samples: int = 1000,
    calibration_function: CalibFuncType = expected_calibration_error,
) -> np.ndarray:
    """
    Calculate the score calibration function from a finite set of confidence predictions which are perfectly calibrated.

    This gives an idea of the range of scores that can still be calibrated,
    used in the null hypothesis test introduced in (https://arxiv.org/pdf/1902.06977.pdf)

    :param conf_preds: The predicted confidence for the point prediction
    :param samples: The number of bootstrap samples
    :param calibration_function: The calibration metric that we want to test]

    :return: n evaluations of a calibration from a perfectly calibrated estimator
    """
    calibrated_scores = np.zeros(samples)
    probs = np.array(conf_preds)
    for i in range(samples):
        calibrated_predictions = consistency_sample(probs)

        # line below is hack so that I don't have to chance calibration functions, calibrated predictions are
        # correct if 1 so the y_true is an array of ones meaning that they match i.e. are counted as correct whenever
        # calibrated_predictions are 1
        calibrated_true = np.ones(len(calibrated_predictions))

        calibrated_scores[i] = calibration_function(
            calibrated_true, calibrated_predictions, probs, 10, 1)

    return calibrated_scores


def consistency_sample(conf_preds: np.ndarray) -> np.ndarray:
    """
    Take a set of confidence scores and returns a set of predictions which are well calibrated.

    These are used to perform null hypothesis testing for the hypothesis of a calibrated forecast
    (Broker, Increasing the Reliability of Reliability Diagrams: https://journals.ametsoc.org/doi/pdf/10.1175/waf993.1)

    :param conf_preds: The predicted confidence for the point prediction

    :return: n confidence estimate samples from a perfectly calibrated estimator
    """
    uniform_samples = np.random.rand(len(conf_preds))
    choices = uniform_samples < conf_preds
    y = np.zeros_like(conf_preds)
    y[choices] = 1
    return y
