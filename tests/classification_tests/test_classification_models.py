# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
import pytest
import numpy as np
import nmslib

from scipy.sparse import csr_matrix
from scipy.sparse import random as sp_rand

from macest.classification.models import _TrainingHelper, ModelWithConfidence, HnswGraphArgs


class MockPPModel:
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
        return np.random.choice((0, 1), x.shape[0])


@pytest.fixture()
def init_model():
    x_train = np.random.rand(100, 2)
    y_train = np.random.choice((0, 1), 100)
    init_model = ModelWithConfidence(MockPPModel, x_train, y_train)  # noqa
    return init_model


@pytest.fixture()
def training_model(init_model):
    x_cal = np.random.rand(100, 2)
    y_cal = np.random.choice((0, 1), 100)
    train = _TrainingHelper(init_model, x_cal, y_cal)
    return train


@pytest.fixture()
def sparse_init_model():
    n_rows = 10 ** 3
    n_cols = 5 * 10 ** 3

    x_train = csr_matrix(sp_rand(n_rows, n_cols))
    y_train = np.random.choice((0, 1), n_rows)

    neighbour_search_params = HnswGraphArgs(query_args=dict(ef=100),
                                            init_args=dict(method="hnsw",
                                                           space="cosinesimil_sparse",
                                                           data_type=nmslib.DataType.SPARSE_VECTOR))

    sparse_init_model = ModelWithConfidence(MockPPModel,
                                            x_train,
                                            y_train,
                                            search_method_args=neighbour_search_params)  # noqa
    return sparse_init_model


def test_batch_prediction(init_model):
    x_test = np.random.rand(2, 2)
    pred = init_model.predict_proba(x_test)
    assert pred is not None
    assert np.all(np.round(np.sum(pred, axis=1), 8) == 1.0)
    assert np.all(pred <= 1.0)
    assert np.all(pred >= 0.0)


def test_finding_nearest_neighbours_batch(init_model):
    x_test = np.random.rand(2, 2)
    init_model._num_neighbours = 5
    dist, ind, error = init_model.calc_dist_to_neighbours(x_test, cls=0)
    assert dist is not None
    assert ind is not None
    assert dist.shape == (2, 5)
    assert ind.shape == (2, 5)
    assert isinstance(ind[0][0], (int, np.integer))
    assert np.all(dist >= 0)
    assert np.all(dist[:, 1:] >= dist[:, :-1])
    assert np.all(dist >= 0.0)
    assert np.all(error >= 0.0)
    assert np.all(error <= 1.0)

    x_test = np.random.rand(2, 2)
    init_model._num_neighbours = 5
    dist, ind, error = init_model.calc_dist_to_neighbours(x_test, cls=1, )
    assert dist is not None
    assert ind is not None
    assert dist.shape == (2, 5)
    assert ind.shape == (2, 5)
    assert isinstance(ind[0][0], (int, np.integer))
    assert np.all(dist >= 0)
    assert np.all(dist[:, 1:] >= dist[:, :-1])
    assert np.all(dist >= 0.0)
    assert np.all(error >= 0.0)
    assert np.all(error <= 1.0)


def test_loss_func(training_model):
    test_params = (1, 1, 5, 1)
    loss = training_model.loss(test_params)
    assert loss is not None
    assert loss >= 0


def test_training(init_model):
    np.random.seed(0)
    initial_alpha = init_model._alpha
    initial_beta = init_model._beta
    initial_temp = init_model._temp
    args = {"popsize": 2, "maxiter": 3}
    x_cal = np.random.rand(100, 2)
    y_cal = np.random.choice((0, 1), 100)

    init_model.fit(x_cal, y_cal, optimiser_args=args)

    assert init_model._alpha != initial_alpha
    assert init_model._beta != initial_beta
    assert init_model._temp != initial_temp


def test_sparse_prediction(sparse_init_model):
    n_cols = 5 * 10 ** 3
    x_test = csr_matrix(sp_rand(2, n_cols))

    pred = sparse_init_model.predict_proba(x_test)
    assert pred is not None
    assert np.all(np.round(np.sum(pred, axis=1), 8) == 1.0)
    assert np.all(pred <= 1.0)
    assert np.all(pred >= 0.0)


def test_sparse_data_consistency_check():
    n_rows = 10 ** 3
    n_cols = 5 * 10 ** 3

    x_train = csr_matrix(sp_rand(n_rows, n_cols))
    y_train = np.random.choice((0, 1), n_rows)
    with pytest.raises(ValueError):
        ModelWithConfidence(MockPPModel,
                            x_train,
                            y_train)  # noqa


def test_space_data_consistency_check():
    n_rows = 10 ** 3
    n_cols = 5 * 10 ** 3

    x_train = csr_matrix(sp_rand(n_rows, n_cols))
    y_train = np.random.choice((0, 1), n_rows)
    neighbour_search_params = HnswGraphArgs(query_args=dict(ef=100),
                                            init_args=dict(method="hnsw",
                                                           space="cosinesimil",
                                                           data_type=nmslib.DataType.SPARSE_VECTOR))

    with pytest.raises(ValueError):
        ModelWithConfidence(MockPPModel,
                            x_train,
                            y_train,
                            search_method_args=neighbour_search_params)


def test_data_space_consistency_check():
    n_rows = 10 ** 3
    n_cols = 5 * 10 ** 3

    x_train = csr_matrix(sp_rand(n_rows, n_cols))
    y_train = np.random.choice((0, 1), n_rows)
    neighbour_search_params = HnswGraphArgs(query_args=dict(ef=100),
                                            init_args=dict(method="hnsw",
                                                           space="cosinesimil",
                                                           data_type=nmslib.DataType.DENSE_VECTOR))

    with pytest.raises(ValueError):
        ModelWithConfidence(MockPPModel,
                            x_train,
                            y_train,
                            search_method_args=neighbour_search_params)
