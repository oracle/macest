# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
import pytest
import numpy as np
import nmslib
from scipy.sparse import csr_matrix
from scipy.sparse import random as sp_rand
from macest.regression.models import (
    ModelWithPredictionInterval,
    _TrainingHelper,
    SearchBounds,
    HnswGraphArgs
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


@pytest.fixture()
def training_model(init_model):
    X_cal = np.random.rand(10, 2)
    y_cal = np.ones(10)
    bnds = SearchBounds()
    train = _TrainingHelper(init_model, X_cal, y_cal, param_range=bnds)
    return train


@pytest.fixture()
def sparse_init_model():
    n_rows = 10 ** 3
    n_cols = 5 * 10 ** 3

    x_train = csr_matrix(sp_rand(n_rows, n_cols))
    y_train = np.random.choice((0, 1), n_rows)

    neighbour_search_params = HnswGraphArgs(query_kwargs=dict(ef=100),
                                            init_kwargs=dict(method="hnsw",
                                                             space="cosinesimil_sparse",
                                                             data_type=nmslib.DataType.SPARSE_VECTOR))

    sparse_init_model = ModelWithPredictionInterval(MockPPModel,
                                                    x_train,
                                                    y_train,
                                                    search_method_args=neighbour_search_params)  # noqa
    return sparse_init_model


def test_batch_prediction(init_model):
    X_test = np.random.rand(2, 2)
    lower, upper = init_model.predict_interval(X_test).T
    lower = lower[0]
    upper = upper[0]

    assert lower is not None
    assert upper is not None
    assert len(lower) == 2
    assert len(upper) == 2
    assert lower[0] < upper[0]
    assert lower[1] < upper[1]


def test_batch_samples(init_model):
    X_test = np.random.rand(2, 2)
    samples = init_model.sample_prediction(X_test, nsamples=100)
    assert samples is not None
    assert samples.shape == (2, 100)


def test_graph_building(init_model):
    graph = init_model.build_graph()
    assert graph is not None
    assert type(graph) == nmslib.dist.FloatIndex


def test_finding_nearest_neighbours_batch(init_model):
    X_test = np.random.rand(2, 2)
    init_model._num_neighbours = 5
    dist, ind = init_model.calc_nn_dist(X_test)
    assert dist is not None
    assert ind is not None
    assert dist.shape == (2, 5)
    assert ind.shape == (2, 5)
    assert isinstance(ind[0][0], (int, np.integer))
    assert np.all(dist >= 0)
    assert np.all(dist[:, 1:] >= dist[:, :-1])


def test_loss_func(training_model):
    test_params = (1, 1, 5)
    loss = training_model.loss_func(test_params)
    assert loss is not None
    assert loss >= 0


def test_training(init_model):
    initial_alpha = init_model._alpha
    initial_beta = init_model._beta
    X_cal = np.random.rand(10, 2)
    y_cal = np.ones(10)
    args = {"popsize": 2, "maxiter": 2}
    init_model.fit(X_cal, y_cal, optimiser_args=args)

    assert init_model._alpha != initial_alpha
    assert init_model._beta != initial_beta


def test_sparse_prediction(sparse_init_model):
    n_cols = 5 * 10 ** 3
    x_test = csr_matrix(sp_rand(2, n_cols))

    lower, upper = sparse_init_model.predict_interval(x_test).T
    lower = lower[0]
    upper = upper[0]

    assert lower is not None
    assert upper is not None
    assert len(lower) == 2
    assert len(upper) == 2
    assert lower[0] < upper[0]
    assert lower[1] < upper[1]


def test_sparse_data_consistency_check():
    n_rows = 10 ** 3
    n_cols = 5 * 10 ** 3

    x_train = csr_matrix(sp_rand(n_rows, n_cols))
    y_train = np.random.choice((0, 1), n_rows)
    with pytest.raises(ValueError):
        ModelWithPredictionInterval(MockPPModel,
                                    x_train,
                                    y_train, )  # noqa


def test_space_data_consistency_check():
    n_rows = 10 ** 3
    n_cols = 5 * 10 ** 3

    x_train = csr_matrix(sp_rand(n_rows, n_cols))
    y_train = np.random.choice((0, 1), n_rows)
    neighbour_search_params = HnswGraphArgs(query_kwargs=dict(ef=100),
                                            init_kwargs=dict(method="hnsw",
                                                             space="cosinesimil",
                                                             data_type=nmslib.DataType.SPARSE_VECTOR))

    with pytest.raises(ValueError):
        ModelWithPredictionInterval(MockPPModel,
                                    x_train,
                                    y_train,
                                    search_method_args=neighbour_search_params)


def test_data_space_consistency_check():
    n_rows = 10 ** 3
    n_cols = 5 * 10 ** 3

    x_train = csr_matrix(sp_rand(n_rows, n_cols))
    y_train = np.random.choice((0, 1), n_rows)
    neighbour_search_params = HnswGraphArgs(query_kwargs=dict(ef=100),
                                            init_kwargs=dict(method="hnsw",
                                                             space="cosinesimil",
                                                             data_type=nmslib.DataType.DENSE_VECTOR))

    with pytest.raises(ValueError):
        ModelWithPredictionInterval(MockPPModel,
                                    x_train,
                                    y_train,
                                    search_method_args=neighbour_search_params)
