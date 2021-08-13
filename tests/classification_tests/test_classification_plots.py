# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
import pytest
import numpy as np
from pathlib import Path
import os

from macest.classification.models import ModelWithConfidence
from macest.classification.plots import (
    plot_calibration_curve,
    plot_quantile_spaced_calibration_curve,
    plot_calibration_metrics,
    plot_confidence_distribution,
    plot_forecast_metrics,
)


class MockPPModel(object):
    """Class to generate random predictions for testing."""

    def __init__(self, x):
        """
        Init.

        :param x: Mock features to be used for prediction
        """
        self.x = x

    @staticmethod
    def predict(x):
        """Return a random prediction to test downstream infrastructure."""
        return np.random.choice((0, 1), len(x))


@pytest.fixture()
def init_model():
    x_train = np.random.rand(100, 2)
    y_train = np.random.choice((0, 1), 100)
    init_model = ModelWithConfidence(MockPPModel, x_train, y_train)  # noqa
    return init_model


def test_plot_calibration_curve(init_model):
    x_star = np.random.rand(100, 2)
    point_preds = init_model.predict(x_star)
    conf_score = init_model.predict_confidence_of_point_prediction(x_star)
    plot_calibration_curve(
        [conf_score],
        labels=["MACEst"],
        point_predictions=point_preds,
        targets=np.random.choice((0, 1), len(x_star)),
        save=True,
        save_dir="./tmp_cal.png",
    )
    assert Path("./tmp_cal.png").is_file()
    os.remove("./tmp_cal.png")


def test_plot_quantile_spaced_calibration_curve(init_model):
    x_star = np.random.rand(100, 2)
    point_preds = init_model.predict(x_star)
    conf_score = init_model.predict_confidence_of_point_prediction(x_star)
    plot_quantile_spaced_calibration_curve(
        [conf_score],
        labels=["MACEst"],
        point_predictions=point_preds,
        targets=np.random.choice((0, 1), len(x_star)),
        save=True,
        save_dir="./tmp_quantile_cal.png",
    )
    assert Path("./tmp_quantile_cal.png").is_file()
    os.remove("./tmp_quantile_cal.png")


def test_plot_calibration_metrics(init_model):
    x_star = np.random.rand(100, 2)
    point_preds = init_model.predict(x_star)
    conf_score = init_model.predict_confidence_of_point_prediction(x_star)
    plot_calibration_metrics(
        [conf_score],
        labels=["MACEst"],
        point_predictions=point_preds,
        targets=np.random.choice((0, 1), len(x_star)),
        save=True,
        save_dir="./tmp_cal_met.png",
    )
    assert Path("./tmp_cal_met.png").is_file()
    os.remove("./tmp_cal_met.png")


def test_plot_forecast_metrics(init_model):
    x_star = np.random.rand(100, 2)
    point_preds = init_model.predict(x_star)
    conf_score = init_model.predict_confidence_of_point_prediction(x_star)
    plot_forecast_metrics(
        [conf_score],
        labels=["MACEst"],
        point_predictions=point_preds,
        targets=np.random.choice((0, 1), len(x_star)),
        save=True,
        save_dir="./tmp_forc_met.png",
    )
    assert Path("./tmp_forc_met.png").is_file()
    os.remove("./tmp_forc_met.png")


def test_plot_confidence_distribution(init_model):
    x_star = np.random.rand(100, 2)
    point_preds = init_model.predict(x_star)
    conf_score = init_model.predict_confidence_of_point_prediction(x_star)
    plot_confidence_distribution(
        conf_score,
        point_predictions=point_preds,
        targets=np.random.choice((0, 1), len(x_star)),
        save=True,
        save_dir="./tmp_conf_dist.png",
    )
    assert Path("./tmp_conf_dist.png").is_file()
    os.remove("./tmp_conf_dist.png")
