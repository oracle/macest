# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from macest.classification.metrics import (
    average_calibration_error,
    expected_calibration_error,
    consistency_sample,
    simulate_from_calibrated_model,
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


def test_average_calibration_error(create_binary_predictions):
    true_values, predictions, confidence, = create_binary_predictions

    ace = average_calibration_error(
        predictions, true_values, confidence, n_bins=10, min_bin_size=1
    )

    frac_correct = np.array([0.5, 1, 4.0 / 6, 4.0 / 5.0, 1.0])
    av_bin_conf = np.array([0.51, 0.62, 0.745, 0.826, 0.93666667])

    true_ace = 100 * abs(frac_correct - av_bin_conf).mean()

    np.testing.assert_approx_equal(ace, true_ace)


def test_expected_calibration_error(create_binary_predictions):
    true_values, predictions, confidence, = create_binary_predictions

    ece = expected_calibration_error(
        predictions, true_values, confidence, n_bins=10, min_bin_size=1
    )

    frac_correct = np.array([0.5, 1, 4.0 / 6, 4.0 / 5.0, 1.0])
    av_bin_conf = np.array([0.51, 0.62, 0.745, 0.826, 0.93666667])
    count = np.array([2.0, 1.0, 6.0, 5.0, 6.0])

    true_ece = (abs(frac_correct - av_bin_conf).dot(count)) / count.sum() * 100

    np.testing.assert_approx_equal(ece, true_ece)


def test_consistency_sample(create_binary_predictions):
    np.random.seed(0)
    true_values, predictions, confidence, = create_binary_predictions
    calibrated_samples = consistency_sample(confidence)
    true_calibrated_samples = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
        ]
    )

    np.testing.assert_array_equal(calibrated_samples, true_calibrated_samples)


def test_simulate_from_calibrated_model(create_binary_predictions):
    np.random.seed(0)
    true_values, predictions, confidence, = create_binary_predictions

    calibrated_eces = simulate_from_calibrated_model(confidence, 5)
    true_eces = np.array([13.2, 21.9, 10.9, 11.8, 12.2])

    np.testing.assert_array_almost_equal(calibrated_eces, true_eces)
