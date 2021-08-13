# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder

from macest.classification.utils import (
    histogram_max_conf_pred,
    calculate_brier_decomposition,
)


@pytest.fixture()
def create_binary_predictions():
    true = np.array(
        [
            "A",
            "B",
            "B",
            "B",
            "B",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "A",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "A",
        ]
    )

    predictions = np.array(
        [
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
        ]
    )

    confidence = np.array(
        (
            [0.91, 0.09],
            [0.18, 0.82],
            [0.81, 0.19],
            [0.17, 0.83],
            [0.51, 0.49],
            [0.29, 0.71],
            [0.71, 0.29],
            [0.05, 0.95],
            [0.95, 0.05],
            [0.18, 0.82],
            [0.91, 0.09],
            [0.21, 0.79],
            [0.75, 0.25],
            [0.15, 0.85],
            [0.51, 0.49],
            [0.38, 0.62],
            [0.72, 0.28],
            [0.05, 0.95],
            [0.95, 0.05],
            [0.21, 0.79],
        )
    )

    enc = LabelEncoder()
    enc.fit(true)
    predictions = enc.transform(predictions)
    true = enc.transform(true)

    prediction_confidence = confidence[np.arange(len(confidence)), predictions]

    return true, predictions, prediction_confidence


@pytest.fixture()
def create_multi_predictions():
    true = np.array(
        [
            "A",
            "B",
            "B",
            "B",
            "B",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "A",
            "A",
            "B",
            "A",
            "B",
            "A",
            "A",
            "A",
            "C",
        ]
    )

    predictions = np.array(
        [
            "A",
            "B",
            "A",
            "B",
            "C",
            "A",
            "A",
            "B",
            "B",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "A",
            "A",
            "A",
        ]
    )

    confidence = np.array(
        (
            [0.48, 0.21, 0.31],
            [0.1, 0.82, 0.08],
            [0.81, 0.1, 0.09],
            [0.1, 0.83, 0.06],
            [0.2, 0.29, 0.41],
            [0.61, 0.20, 0.09],
            [0.7, 0.2, 0.1],
            [0.03, 0.95, 0.02],
            [0.01, 0.98, 0.01],
            [0.01, 0.89, 0.1],
            [0.91, 0.05, 0.05],
            [0.18, 0.79, 0.03],
            [0.71, 0.24, 0.05],
            [0.13, 0.85, 0.01],
            [0.51, 0.48, 0.01],
            [0.31, 0.62, 0.07],
            [0.72, 0.18, 0.1],
            [0.94, 0.02, 0.04],
            [0.96, 0.02, 0.02],
            [0.34, 0.33, 0.33],
        )
    )

    enc = LabelEncoder()
    enc.fit(true)
    predictions = enc.transform(predictions)
    true = enc.transform(true)

    prediction_confidence = confidence[np.arange(len(confidence)), predictions]
    return true, predictions, prediction_confidence


def test_uniform_histogram_binary(create_binary_predictions):
    true_values, predictions, confidence, = create_binary_predictions

    bins, frac_correct, count, av_conf, _ = histogram_max_conf_pred(
        true_values, predictions, confidence, bin_method="uniform", min_bin_size=1
    )
    assert np.all(np.round(np.diff(bins), 1) == 0.1)
    assert np.all(frac_correct >= 0)
    assert np.all(frac_correct <= 1)
    assert np.all(count >= 0)
    np.testing.assert_array_equal(count, np.array([2.0, 1.0, 6.0, 5.0, 6.0]))
    np.testing.assert_array_equal(
        frac_correct, np.array([0.5, 1, 4.0 / 6, 4.0 / 5.0, 6.0 / 6.0])
    )


def test_uniform_histogram_multi(create_multi_predictions):
    true_values, predictions, confidence, = create_multi_predictions

    bins, frac_correct, count, av_conf, _ = histogram_max_conf_pred(
        true_values, predictions, confidence, bin_method="uniform", min_bin_size=1
    )

    assert np.all(frac_correct >= 0)
    assert np.all(frac_correct <= 1)
    assert np.all(count >= 0)
    np.testing.assert_array_equal(count, np.array([1.0, 2, 1.0, 3.0, 3.0, 5.0, 5.0]))
    np.testing.assert_array_equal(
        frac_correct,
        np.array([0.0 / 1, 1.0 / 2, 1.0 / 1, 3.0 / 3, 2.0 / 3, 4.0 / 5.0, 5.0 / 5]),
    )


def test_bin_size_param(create_multi_predictions):
    true_values, predictions, confidence, = create_multi_predictions

    bins, frac_correct, count, av_conf, _ = histogram_max_conf_pred(
        true_values,
        predictions,
        confidence,
        bin_method="uniform",
        merge_bins_below_threshold=False,
        min_bin_size=4,
    )
    assert np.all(frac_correct >= 0)
    assert np.all(frac_correct <= 1)
    assert np.all(count >= 0)
    np.testing.assert_array_equal(count, np.array([5.0, 5.0]))
    np.testing.assert_array_equal(frac_correct, np.array([4.0 / 5.0, 1]))


def test_bin_merge(create_multi_predictions):
    true_values, predictions, confidence, = create_multi_predictions

    bins, frac_correct, count, av_conf, _ = histogram_max_conf_pred(
        true_values,
        predictions,
        confidence,
        bin_method="uniform",
        merge_bins_below_threshold=True,
        min_bin_size=3,
    )
    assert np.all(frac_correct >= 0)
    assert np.all(frac_correct <= 1)
    assert np.all(count >= 0)
    assert count.sum() == len(confidence)
    np.testing.assert_array_equal(count, np.array([3, 4, 3, 5.0, 5.0]))
    np.testing.assert_array_equal(
        frac_correct, np.array([1.0 / 3.0, 4.0 / 4.0, 2.0 / 3.0, 4.0 / 5.0, 1])
    )


def test_brier_decomposition(create_binary_predictions):
    true_values, predictions, confidence, = create_binary_predictions

    bs = calculate_brier_decomposition(confidence, predictions, true_values)

    true_bs = np.array(
        (0.010443166666666667, 0.028333333333333332, 0.15999999999999998)
    )

    np.testing.assert_array_equal(bs, true_bs)
