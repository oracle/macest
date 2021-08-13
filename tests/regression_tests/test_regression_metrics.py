# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
import pytest
import numpy as np

from macest.regression.models import (
    ModelWithPredictionInterval,
    picp_loss,
)
from macest.regression.metrics import (
    predictions_in_range,
    mean_prediction_interval_width,
)


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
        return 1.0


@pytest.fixture()
def init_model():
    train_error = np.ones(100)
    x_train = np.random.rand(100, 2)
    init_model = ModelWithPredictionInterval(
        model=MockPPModel, x_train=x_train, train_err=train_error
    )
    return init_model


def test_picp(init_model):
    x_test = np.random.rand(100, 2)
    y_test = np.random.uniform(0, 1, 100)
    loss = picp_loss(init_model, x_test, y_test)
    assert loss is not None
    assert loss >= 0


def test_mpiw(init_model):
    x_test = np.random.rand(100, 2)
    mean_width = mean_prediction_interval_width(init_model, x_test)
    assert mean_width is not None
    assert mean_width >= 0


def test_predictions_inrange(init_model):
    x_test = np.random.rand(100, 2)
    y_test = np.random.uniform(0, 1, 100)
    inrange = predictions_in_range(
        y_test, x_test, init_model, conf_level=90, verbose=False
    )
    assert inrange is not None
    assert np.all(inrange >= 0)


# add use case where I know the loss and the inrange values for each interval and test that
# I return these
