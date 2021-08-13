# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
"""Utils module for random function used by MACEst in classification."""

import logging
import numpy as np
from typing import Optional, Union, Sequence, NamedTuple

from typing_extensions import Literal

log = logging.getLogger()


class HistogramPredsOutput(NamedTuple):
    """Container for the binned confidence predictions information."""

    bins: np.ndarray
    frac_correct: np.ndarray
    bin_count: np.ndarray
    av_conf: np.ndarray
    conflicting_predictions: Union[np.ndarray, None]


class BrierDecomposition(NamedTuple):
    """Container for three part brier composition."""

    calibration: float
    resolution: float
    uncertainty: float


class MergedBinData(NamedTuple):
    """Container for information about bins after adjusting bins to make enough samples."""

    bin_count: np.ndarray
    bin_confidence_sum: np.ndarray
    bin_num_correct: np.ndarray


def histogram_max_conf_pred(
        targets: Union[Sequence[int], np.ndarray],
        predictions: Union[Sequence[int], np.ndarray],
        point_prediction_conf: Union[Sequence[int], np.ndarray],
        class_conf: Optional[Union[Sequence[float], np.ndarray]] = None,
        n_bins: int = 10,
        min_bin_size: int = 10,
        merge_bins_below_threshold: bool = True,
        bin_method: Literal["quantile", "uniform"] = "quantile",
        check_conflicting_preds: bool = False,
) -> HistogramPredsOutput:
    """
    Take a set of confidence estimates and groups them into bins of similar confidence, then
        calculate the average confidence and accuracy fot that bin. This can then be used for
        calibration metrics and plots.

    :param targets: The true target y values
    :param predictions: The model point predictions
    :param point_prediction_conf: The confidence associated with those point predictions
    :param class_conf: The original MACEst confidence estimate all each class
    :param min_bin_size: the minimum number of elements in a min before it is merged with the one
        to the right
    :param merge_bins_below_threshold: If true merge bins with counts below the min size, if false,
        discard bin below threshold, note this over-rides n_bins
    :param bin_method: the method to group the confidence estimates into emperical bins, currently
        only uniform or quantile binning implemented.
    :param check_conflicting_preds: If true this will signal that the max MACEst confidence
        prediction and the point prediction do not agree, it will then print the length and the
        index for these conflict
    :param n_bins: The number of bins to use for binning the confidence scores, must be bigger than
        0

    :return: bin intervals, average accuracy per bin, number of elements per bin, average
        confidence per bin
    """
    conflicting_predictions = None
    if check_conflicting_preds:
        conflicting_predictions = check_for_conflicting_preds(class_conf, predictions)

    if bin_method == "quantile":
        quantile_increments = 1.0 / n_bins
        bins = np.quantile(
            point_prediction_conf,
            np.arange(0.0, 1.01, quantile_increments),
            interpolation="nearest",
        )
    elif bin_method == "uniform":
        step_size = 1.0 / n_bins
        bins = np.arange(step_size, 1 + step_size / 100000, step_size)
    # provide elif option to give bins as a fixed array/list
    else:
        raise ValueError("Only quantile and uniform binning method implemented")

    bin_idxs = np.digitize(point_prediction_conf, bins, right=True)

    prediction_is_correct = targets == predictions

    bin_count = np.bincount(bin_idxs, minlength=len(bins))
    bin_conf_sum = np.bincount(
        bin_idxs, weights=point_prediction_conf, minlength=len(bins)
    )
    bin_num_correct = np.bincount(
        bin_idxs, weights=prediction_is_correct, minlength=len(bins)
    )

    if bin_method == "uniform" and merge_bins_below_threshold:
        bin_count, bin_conf_sum, bin_num_correct = _merge_low_count_bins(
            bin_count, bin_conf_sum, bin_num_correct, min_bin_size
        )

    bins_to_keep = bin_count >= min_bin_size
    bin_count = bin_count[bins_to_keep]
    frac_correct = bin_num_correct[bins_to_keep] / bin_count
    av_conf = bin_conf_sum[bins_to_keep] / bin_count

    return HistogramPredsOutput(bins, frac_correct, bin_count, av_conf, conflicting_predictions)


def _merge_low_count_bins(
        bin_count: np.ndarray,
        bin_confidence_sum: np.ndarray,
        bin_num_correct: np.ndarray,
        min_bin_size: int,
) -> MergedBinData:
    """
    Merge the bin with the bin to the right, If the number of elements in a bin is below a certain threshold, \
    this will be an iterable procedure until every bin is over the desired threshold, this over-rides \
    the n_bins.

    :param bin_count: number of predictions in confidence bin
    :param bin_confidence_sum: total
    :param bin_num_correct: number of correct predictions in each bin
    :param min_bin_size: min allowed number of predictions in confidence bin, if below this merge to the right
    :return: MergedBinData, this contains - bin_count, bin_confidence_sum, bin_num_correct
    """
    all_bins_over_min = False
    idx = 0
    while not all_bins_over_min:
        if bin_count[idx] < min_bin_size:
            bin_count[idx + 1] = bin_count[idx] + bin_count[idx + 1]
            bin_confidence_sum[idx + 1] = (
                    bin_confidence_sum[idx] + bin_confidence_sum[idx + 1]
            )
            bin_num_correct[idx + 1] = bin_num_correct[idx] + bin_num_correct[idx + 1]
            bin_count[idx] = -1
            bin_confidence_sum[idx] = -1
            bin_num_correct[idx] = -1

        bin_count = bin_count[bin_count > -1]
        bin_confidence_sum = bin_confidence_sum[bin_confidence_sum != -1]
        bin_num_correct = bin_num_correct[bin_num_correct != -1]

        if bin_count[idx] >= min_bin_size:
            idx += 1

        if all(bin_count >= min_bin_size) or idx == len(bin_count) - 1:
            all_bins_over_min = True

    return MergedBinData(bin_count, bin_confidence_sum, bin_num_correct)


def calculate_brier_decomposition(
        confidence_scores: np.ndarray,
        point_preds: np.ndarray,
        targets: np.ndarray,
        min_bin_size: int = 1,
) -> BrierDecomposition:
    """
    Calculate the three-part decomposition of the brier score, this allows one to identify different important \
    aspects of a forecast.

    Full brier score = CAl - RES + UNC.

    :param confidence_scores: the predicted confidence for the point prediction
    :param targets: The true target y values
    :param point_preds: The model point predictions
    :param min_bin_size: The threshold number of elements in a bin
    :return: calibration, resolution, uncertainty
    """
    prediction_is_correct = point_preds == targets

    bins = np.arange(0, 1.01, 0.1)

    bin_idxs = np.digitize(confidence_scores, bins, right=True)

    bin_count = np.bincount(bin_idxs, minlength=len(bins))
    bin_conf_sum = np.bincount(bin_idxs, weights=confidence_scores, minlength=len(bins))
    bin_num_correct = np.bincount(
        bin_idxs, weights=prediction_is_correct, minlength=len(bins)
    )

    bin_count, bin_conf_sum, bin_num_correct = _merge_low_count_bins(
        bin_count, bin_conf_sum, bin_num_correct, min_bin_size
    )

    bins_to_keep = bin_count >= min_bin_size
    bin_count = bin_count[bins_to_keep]
    frac_correct = bin_num_correct[bins_to_keep] / bin_count
    av_conf = bin_conf_sum[bins_to_keep] / bin_count

    calibration = (
                      (((frac_correct - av_conf) ** 2) * bin_count).sum()
                  ) / bin_count.sum()
    uncertainty = prediction_is_correct.mean() * (1 - prediction_is_correct.mean())

    resolution = (
                     ((frac_correct - prediction_is_correct.mean()) ** 2 * bin_count).sum()
                 ) / bin_count.sum()

    return BrierDecomposition(calibration, resolution, uncertainty)


def check_for_conflicting_preds(class_conf: Union[Sequence[int], np.ndarray],
                                predictions: Union[Sequence[int], np.ndarray]) -> np.ndarray:
    """
    Check when MACEst max confidence prediction is different to the point prediction.

    :param predictions: The model point predictions
    :param class_conf: The original MACEst confidence estimate all each class
    :return:
    """
    max_confidence_prediction = np.argmax(class_conf, axis=1)
    conflicting_predictions = np.argwhere(
        max_confidence_prediction != predictions
    ).flatten()
    num_conflicting = len(conflicting_predictions) / len(max_confidence_prediction)
    log.info(
        f"{num_conflicting * len(max_confidence_prediction)} ({num_conflicting * 100:.2f}%) max "
        f"confidence estimates do not agree with the ml point prediction"
    )
    return conflicting_predictions
