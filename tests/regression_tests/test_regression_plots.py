# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
import pytest
import numpy as np
from macest.regression.models import ModelWithPredictionInterval, _TrainingHelper
from macest.regression.plots import (
    plot_pred_with_conf,
    plot_calibration,
    plot_true_vs_predicted,
    plot_predicted_vs_true,
)
from pathlib import Path
import os


class MockPPModel:
    """Class to generate constant predictions for testing."""

    def __init__(self, x):
        """
        Init.

        :param x: Mock features to be used for prediction
        """
        self.x = x

    def predict(x):
        """Return a constant prediction to test downstream infrastructure."""
        return np.random.uniform(0, 1, x.shape[0])


@pytest.fixture()
def init_model():
    train_error = np.ones(100)
    x_train = np.random.rand(100, 2)
    init_model = ModelWithPredictionInterval(
        model=MockPPModel, x_train=x_train, train_err=train_error
    )
    return init_model


@pytest.fixture()
def training_model(init_model):
    X_cal = np.random.rand(10, 2)
    y_cal = np.ones(10)
    bnds = ((1.1, 1.5), (1, 1.5), (5, 7))
    train = _TrainingHelper(init_model, X_cal, y_cal, param_range=bnds)
    return train


def test_plot_pred_with_conf(init_model):
    x_star = np.random.rand(1, 2)
    plot_pred_with_conf(init_model, x_star, save=True, save_dir="./tmp_dist.png")
    assert Path("./tmp_dist.png").is_file()
    os.remove("./tmp_dist.png")


def test_plot_calibration(init_model):
    x_star = np.random.rand(100, 2)
    plot_calibration(
        init_model,
        x_star=x_star,
        y_true=np.random.rand(len(x_star)),
        save=True,
        save_dir="./tmp_cal.png",
    )
    assert Path("./tmp_cal.png").is_file()
    os.remove("./tmp_cal.png")


def test_plot_true_vs_actual(init_model):
    x_star = np.random.rand(100, 2)
    plot_true_vs_predicted(
        model=MockPPModel,
        conf_model=init_model,
        x_star=x_star,
        y_true=np.random.rand(len(x_star)),
        save=True,
        save_dir="./tmp_tva.png",
    )
    assert Path("./tmp_tva.png").is_file()
    os.remove("./tmp_tva.png")


def test_plot_actual_vs_true(init_model):
    x_star = np.random.rand(100, 2)
    plot_predicted_vs_true(
        model=MockPPModel,
        conf_model=init_model,
        x_star=x_star,
        y_true=np.random.rand(len(x_star)),
        save=True,
        save_dir="./tmp_avt.png",
    )
    assert Path("./tmp_avt.png").is_file()
    os.remove("./tmp_avt.png")
