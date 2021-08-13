# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
"""Module contains the MACEst model to estimate prediction intervals."""

import nmslib
import logging
import os
import numpy as np
import scipy
from scipy.optimize import differential_evolution
from scipy.stats import laplace, norm
from scipy.stats._continuous_distns import laplace_gen
from typing import Optional, Union, NamedTuple, Tuple, Dict, Any
from typing_extensions import Protocol, Literal

log = logging.getLogger()

num_threads_available = os.cpu_count()


class _RegressionPointPredictionModel(Protocol):
    """
    Defines a protocol for the type PointPredModel.

    if a model has a predict method then it is the same type as PointPredModel.
    """

    def predict(self, x_star: np.ndarray) -> Any:
        """Return nothing as is only needed to check method exists."""
        pass


class HnswGraphArgs(NamedTuple):
    """Object for passing arguments to the nmslib function."""

    init_kwargs: Dict[str, str] = {"method": "hnsw", "space": "l2"}
    construction_kwargs: Dict[str, int] = {"post": 2, "efConstruction": 1000, "M": 100}
    query_kwargs: Dict[str, int] = {"ef": 1000}


class SearchBounds(NamedTuple):
    """Object for passing the range of allowed MACEst parameters."""

    alpha_bounds: Tuple[float, float] = (0.1, 50.0)
    beta_bounds: Tuple[float, float] = (0.1, 50.0)
    k_bounds: Tuple[int, int] = (5, 20)


class MacestPredIntervalModelParams(NamedTuple):
    """Class container for MACEst model parameters."""

    alpha: float = 1.0
    beta: float = 1.0
    num_neighbours: int = 10


class PrecomputedNeighbourInfo(NamedTuple):
    """Class container for the information about pre-computed nearest neighbours per class."""

    prec_distance_to_nn: Union[Dict[int, np.ndarray], np.ndarray]
    prec_ind_of_nn: Union[Dict[int, np.ndarray], np.ndarray]  # Rhys- only dict for training, where I cache
    # arrays for all values of allowed num neighbours


class ModelWithPredictionInterval:
    """Creates a model which returns a prediction and a confidence interval."""

    def __init__(
        self,
        model: _RegressionPointPredictionModel,
        x_train: np.ndarray,
        train_err: np.ndarray,
        macest_model_params: MacestPredIntervalModelParams = MacestPredIntervalModelParams(),
        error_dist: Literal["normal", "laplace"] = "normal",
        dist_func: Literal["linear", "error_weighted_poly"] = "linear",
        precomputed_neighbour_info: Optional[PrecomputedNeighbourInfo] = None,
        prec_point_preds: Optional[np.ndarray] = None,
        prec_graph: Optional[nmslib.dist.FloatIndex] = None,
        search_method_args: HnswGraphArgs = HnswGraphArgs(),
    ):
        """
        Init.

        :param model: Any model which takes some variables x and returns a point prediction y
        :param x_train: The variables used to train the model
        :param train_err: The error for each training point
        :param num_neighbours: The number of points which define the local neighbourhood
        :param alpha: co-efficient for distance function (hyper-parameter)
        :param beta: The hyper-parameter used in distance function
        :param error_dist: The assumed distribution for the errors
        :param dist_func: The function to convert distance to confidence \
                          (currently linear or error_weighted_poly implemented)
        :param prec_point_preds: The pre-computed model predictions
        :param prec_distance_to_nn: The pre-computed nearest neighbour distances for the calibration and test data
        :param prec_ind_of_nn: The pre-computed nearest neighbour indices for the calibration and test data
        :param prec_graph: The pre-computed graph to use for online hnsw search
        """
        self.model = model
        self.x_train = x_train
        self.train_err = train_err
        self.macest_model_params = macest_model_params
        self._num_neighbours = macest_model_params.num_neighbours
        self._alpha = macest_model_params.alpha
        self._beta = macest_model_params.beta
        self.dist_func = dist_func
        self.error_dist = error_dist
        self.prec_graph = prec_graph
        self.point_preds = prec_point_preds
        self.precomputed_neighbour_info = precomputed_neighbour_info
        if not self.precomputed_neighbour_info:
            self._distance_to_nn = None
            self._ind_of_nn = None
        else:
            self._distance_to_nn = self.precomputed_neighbour_info.prec_distance_to_nn
            self._ind_of_nn = self.precomputed_neighbour_info.prec_ind_of_nn
        self.search_method_args = search_method_args
        self._check_consistent_search_method_args()
        self._check_data_consistent_with_search_args()

    def predict(self, x_star: np.ndarray) -> np.ndarray:
        """
        Return a point prediction for x_star.

        :param x_star: The position for which we would like to predict

        :return: pred_star : The point prediction for x_star
        """
        pred_star = self.model.predict(x_star)
        return pred_star

    def build_graph(self) -> nmslib.dist.FloatIndex:
        """
        Build the  Hierarchical Navigable Small World (hnsw) index graph.

        :return: A queryable HNSW graph
        """
        graph = nmslib.init(**self.search_method_args.init_kwargs)
        graph.addDataPointBatch(self.x_train)
        graph.createIndex(self.search_method_args.construction_kwargs)
        graph.setQueryTimeParams(self.search_method_args.query_kwargs)

        return graph

    def calc_nn_dist(self, x_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the distant to a set of k nearest neighbours.

        :param x_star: The position for which we would like to predict

        :return: The distance to k nearest neighbours and the indices of the k closest neighbours
        """
        if self.prec_graph is None:
            self.prec_graph = self.build_graph()

        neighbours = np.array(
            self.prec_graph.knnQueryBatch(
                x_star, k=self._num_neighbours, num_threads=num_threads_available
            )
        )
        dist = neighbours[:, 1, :]
        ind = neighbours[:, 0, :].astype(int)
        return dist, ind

    def calc_linear_dist_func(self, x_star: np.ndarray) -> np.ndarray:
        """
        Calculate the linear sum of average distance to neighbours and average per neighbour error.

        :param x_star: The position for which we would like to predict

        :return: the sum of average distance to neighbours and average per neighbour error for x_star
        """
        if self._distance_to_nn is not None:
            local_distance = self._distance_to_nn
            if self._ind_of_nn is None:
                raise ValueError("_ind_of_nn has not been cached during training")
            ind = self._ind_of_nn
        else:
            local_distance, ind = self.calc_nn_dist(x_star)
        if isinstance(local_distance, np.ndarray):
            dist = self._alpha * np.average(
                local_distance, weights=np.arange(local_distance.shape[1], 0, -1), axis=1,
            )
        else:
            raise ValueError('Need to remove pre-cached training neighbour data from training')
        if isinstance(ind, np.ndarray):
            error = self._beta * np.average(
                abs(self.train_err[ind.astype(int)]),
                weights=1.0 / (1 + local_distance),
                axis=1,)
        else:
            raise ValueError('Need to remove pre-cached training neighbour data from training')

        return dist + error

    def calc_error_weighted_dist(self, x_star: np.ndarray) -> np.ndarray:
        """
        Calculate average distance to neighbours weighted by the per neighbour prediction error.

        :param x_star: The position for which we would like to predict

        :return: the error weighted distance from x_star point to it's neighbours
        """
        if self._distance_to_nn is not None:
            local_distance = self._distance_to_nn
            if self._ind_of_nn is None:
                raise ValueError("_ind_of_nn has not been cached during training")
            ind = self._ind_of_nn
        else:
            local_distance, ind = self.calc_nn_dist(x_star)

        if isinstance(ind, np.ndarray):
            train_error = self.train_err[ind.astype(int)]
        else:
            raise ValueError('Need to remove pre-cached training neighbour data from training')
        if isinstance(local_distance, np.ndarray):
            error_weighted_dist = np.average(
                local_distance * abs(train_error),
                weights=1.0 / (1 + local_distance),
                axis=1,
            )
        else:
            raise ValueError('Need to remove pre-cached training neighbour data from training')

        error_weighted_poly = self._alpha * error_weighted_dist ** self._beta
        return error_weighted_poly

    def std_on_y_star(self, x_star: np.ndarray) -> np.ndarray:
        """
        Return the predicted variance for x_star.

        :param x_star: The position for which we would like to predict

        :return: sigma: The standard deviation for the prediction at x_star
        """
        if self.dist_func == "error_weighted_poly":
            dist = self.calc_error_weighted_dist(x_star)
        elif self.dist_func == "linear":
            dist = self.calc_linear_dist_func(x_star)
        else:
            raise ValueError(f"Unknown distance function: {self.dist_func}")
        sigma = dist
        return sigma

    def laplace_scale_on_y_star(self, x_star: np.ndarray) -> np.ndarray:
        """
        Return the predicted laplacian variance for x_star.

        :param x_star: The position for which we would like to predict

        :return: sigma: The laplacian scaler for the prediction at x_star
        """
        if self.dist_func == "error_weighted_poly":
            dist = self.calc_error_weighted_dist(x_star)
        elif self.dist_func == "linear":
            dist = self.calc_linear_dist_func(x_star)
        else:
            raise ValueError(f"Unknown distance function: {self.dist_func}")
        sigma = dist
        return sigma

    def _distribution(self, x_star: np.ndarray) -> laplace_gen:
        """
        Return the distribution that we will predict from.

        :return:
        """
        if self.point_preds is not None:
            point_preds = self.point_preds
        else:
            point_preds = self.predict(x_star,)
        if self.error_dist == "normal":
            scale = self.std_on_y_star(x_star,)
            dist = norm(loc=point_preds, scale=scale)
        elif self.error_dist == "laplace":
            scale = self.laplace_scale_on_y_star(x_star,)
            dist = laplace(loc=point_preds, scale=scale)
        else:
            raise ValueError(f"Unknown distance function: {self.dist_func}")
        return dist

    def predict_interval(
        self, x_star: np.ndarray, conf_level: Union[np.ndarray, int, float] = 90,
    ) -> np.ndarray:
        """
        Predict the upper and lower prediction interval bounds for a given confidence level.

        :param x_star: The position for which we would like to predict
        :param conf_level:

        :return: The confidence bounds for each x_star for each confidence level
        """
        dist = self._distribution(x_star)
        lower_perc = (100 - conf_level) / 2
        upper_perc = 100 - lower_perc

        lower_vec = 0.01 * np.ones((x_star.shape[0], len([conf_level]))) * lower_perc
        upper_vec = 0.01 * np.ones((x_star.shape[0], len([conf_level]))) * upper_perc
        return np.array([dist.ppf(lower_vec.T), dist.ppf(upper_vec.T)]).T

    def calculate_prediction_interval_width(
        self, x_star: np.ndarray, conf_level: Union[np.ndarray, int, float] = 90,
    ) -> np.ndarray:
        """
        Calculate the absolute width of a prediction interval for a given confidence level.

        :param x_star: The position for which we would like to predict
        :param conf_level:

        :return: the absolute width of a prediction interval for each x_star for each confidence level
        """
        intervals = self.predict_interval(x_star, conf_level)
        return np.diff(intervals)

    def sample_prediction(
        self, x_star: np.ndarray, nsamples: int = 10 ** 3
    ) -> np.ndarray:
        """
        Draw samples from any predicted distribution to get a distribution of predictions.

        :param x_star: The position in feature space for which we would like to predict
        :param nsamples: The number of samples to draw from the distribution

        :return: Samples from the predicted distribution
        """
        dist = self._distribution(x_star)
        return dist.rvs(size=(nsamples, x_star.shape[0])).T

    def fit(
        self,
        x_cal: np.ndarray,
        y_cal: np.ndarray,
        param_range: SearchBounds = SearchBounds(),
        optimiser_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """
        Fit MACEst model using the calibration data.

        :param x_cal: Calibration data
        :param y_cal: Target values
        :param param_range: The bounds within which to search for MACEst parameters
        :param optimiser_args: Any arguments for the optimiser (see scipy.optimize)

        :return: None
        """
        if optimiser_args is None:
            optimiser_args = {}

        train_helper = _TrainingHelper(self, x_cal, y_cal, param_range)
        train_helper.fit(optimiser_args=optimiser_args)

    def _check_consistent_search_method_args(self) -> None:
        init_args = self.search_method_args.init_kwargs
        index = nmslib.init(**init_args)

        if 'space' not in list(init_args.keys()):
            raise ValueError('You must pass a space in your search method init args')

        space = init_args['space']
        if space[-6:] == 'sparse':
            sparse_metric = True
        else:
            sparse_metric = False

        data_type = index.dataType
        if data_type == nmslib.DataType.SPARSE_VECTOR:
            sparse_data = True
        else:
            sparse_data = False

        if sparse_metric != sparse_data:
            raise ValueError(
                f'Data type and space are not compatible, your space is {space} '
                f'and search data type is data_type nmslib.{data_type}')

    def _check_data_consistent_with_search_args(self) -> None:
        init_args = self.search_method_args.init_kwargs

        space = init_args['space']
        if space[-6:] == 'sparse':
            sparse_metric = True
        else:
            sparse_metric = False

        training_data_type = type(self.x_train)

        if training_data_type == scipy.sparse.csr.csr_matrix:
            sparse_data = True
        else:
            sparse_data = False

        if sparse_metric != sparse_data:
            raise ValueError(
                f'Training data type and space are not compatible, your space is {space} '
                f'and training data type is {training_data_type}')


class _TrainingHelper(object):
    def __init__(
        self,
        init_conf_model: ModelWithPredictionInterval,
        x_cal: np.ndarray,
        y_cal: np.ndarray,
        param_range: SearchBounds = SearchBounds(),
    ):
        """
        Init.

        :param init_conf_model: an initialised ModelWithConfidence object that we want to fit
        :param x_cal: The X variables that we will use to calibrate the confidence predictions
        :param y_cal: The target variables that we will use to calibrate the confidence predictions
        :param param_range: The bounds on the hyper-parameter space we want to search
        """
        self.model = init_conf_model
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.param_range = param_range
        self.prec_graph = self.model.build_graph()
        self.model.prec_graph = self.prec_graph
        self.prec_dist, self.prec_ind = self._prec_neighbours()
        self.model.point_preds = self.model.predict(self.x_cal)

    def _prec_neighbours(self) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Pre-compute the nearest neighbours and their distances.

        :return:
        """
        min_nbrs = self.param_range[2][0]
        max_nbrs = self.param_range[2][1]
        num_nbrs = np.arange(min_nbrs, max_nbrs + 0.1, 1)
        x_cal_len_array = np.arange(len(self.x_cal))

        dist_dict = {}
        ind_dict = {}

        max_neighbours = np.array(
            self.prec_graph.knnQueryBatch(
                self.x_cal, k=int(max_nbrs), num_threads=num_threads_available
            )
        )

        max_dist = max_neighbours[x_cal_len_array, 1]
        max_ind = max_neighbours[x_cal_len_array, 0]

        for k in num_nbrs:
            dist = max_dist[x_cal_len_array, 0: int(k)]
            ind = max_ind[x_cal_len_array, 0: int(k)]

            dist_dict[k] = dist
            ind_dict[k] = ind

        return dist_dict, ind_dict

    def set_macest_model_params(self) -> MacestPredIntervalModelParams:
        """
        Return MACEst parameter values.

        :return:
        """
        params = MacestPredIntervalModelParams(
            num_neighbours=self.model._num_neighbours,
            alpha=self.model._alpha,
            beta=self.model._beta,
        )
        self.model.macest_model_params = params
        return params

    def loss_func(self, params: MacestPredIntervalModelParams) -> float:
        """
        Calculate the loss for a given set of parameters, this will then be optimised when fit is called.

        :param params: A tuple containing the model hyper-paramters
        :return:
        """
        self.model._alpha, self.model._beta, self.model._num_neighbours = params

        self.model._num_neighbours = int(np.round(self.model._num_neighbours))

        self.model.prec_graph = self.prec_graph
        self.model._distance_to_nn = self.prec_dist[self.model._num_neighbours]
        self.model._ind_of_nn = self.prec_ind[self.model._num_neighbours]

        return picp_loss(self.model, self.x_cal, self.y_cal)

    def fit(
        self,
        optimiser: Literal["de"] = "de",
        optimiser_args: Optional[Dict[Any, Any]] = None,
    ) -> ModelWithPredictionInterval:
        """
        Fit MACEst parameters.

        :param optimiser: The optimisation method
        :param optimiser_args: Any arguments for the optimisation strategy
        :return: A ModelWithConfidence object with the hyper-parameters that minimises the loss function
        """
        if optimiser == "de":
            result = differential_evolution(
                self.loss_func, self.param_range, **optimiser_args
            )
        else:
            raise ValueError(
                "The only optimisation method currently implemented is differential evolution"
            )

        log.info(f"min_loss = {result.fun}")

        alpha, beta, k = result.x
        k = int(np.round(k, 0))
        log.info(f" best_alpha: {alpha}")
        log.info(f" best_beta: {beta}")
        log.info(f" best_k: {k}")

        self.model._alpha = alpha
        self.model._beta = beta
        self.model._num_neighbours = int(np.round(k))

        self.model.macest_model_params = self.set_macest_model_params()

        self.model._distance_to_nn = None
        self.model._ind_of_nn = None
        self.model.point_preds = None

        return self.model


def picp_loss(
    interval_model: ModelWithPredictionInterval, x_test: np.ndarray, y_true: np.ndarray
) -> float:
    """
    Calculate the difference between the desired confidence level and the \
    prediction_interval_coverage_probability for several intervals.

    :param interval_model: Some model which makes predictions with a standard deviation
    :param x_test: The variables for which we would like to use to predict a distribution
    :param y_true: The True target value

    :return: The loss score
    """
    levels = np.array((90, 70, 50, 30, 10))

    intervals = interval_model.predict_interval(x_test, conf_level=levels,).T

    lower = intervals[0]
    upper = intervals[1]
    loss = 0

    for i in range(len(levels)):
        loss += abs(
            levels[i]
            - (
                100
                * len(
                    np.where(np.logical_and(y_true >= lower[i], y_true <= upper[i]))[0]
                )
                / len(y_true)
            )
        )
    return loss / len(levels)
