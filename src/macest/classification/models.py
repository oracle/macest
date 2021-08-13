# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
"""Module contains the MACEst model to estimate confidence."""
from typing import Iterable, Optional, Dict, NamedTuple, Tuple, Any

import nmslib
import logging
import os
import numpy as np
import scipy
from scipy.optimize import differential_evolution
from scipy.special import softmax
from typing_extensions import Protocol, Literal
from macest.classification.metrics import expected_calibration_error

log = logging.getLogger()

num_threads_available = os.cpu_count()


class _ClassificationPointPredictionModel(Protocol):
    """
    Class which defines a protocol for the type PointPredModel.

    if a model has a predict method then it is the same type as PointPredModel
    """

    def predict(self, x_star: np.ndarray) -> np.ndarray:
        """Return nothing as is only needed to check method exists."""
        pass


class HnswGraphArgs(NamedTuple):
    """Object for passing arguments to the nmslib function."""

    init_args: Dict[str, str] = dict(method="hnsw", space="l2")
    construction_args: Dict[str, int] = dict(post=2, efConstruction=1000, M=100)
    query_args: Dict[str, int] = dict(ef=1000)


class SearchBounds(NamedTuple):
    """Object for passing the range of allowed MACEst parameters."""

    alpha_bounds: Tuple[float, float] = (0.1, 100.0)
    beta_bounds: Tuple[float, float] = (0.1, 100.0)
    k_bounds: Tuple[int, int] = (5, 40)
    temperature_bounds: Tuple[float, float] = (0, 40)


class MacestConfModelParams(NamedTuple):
    """Class container for macest model parameters."""

    alpha: float = 1.0
    beta: float = 1.0
    num_neighbours: int = 10
    temp: float = 1.0


class NearestNeighbourInClassInfo(NamedTuple):
    """Class container for the information about the nearest neighbours to a point in each class."""

    class_dist: np.ndarray
    class_ind: np.ndarray
    class_error: np.ndarray


class PrecomputedNeighbourInfo(NamedTuple):
    """Class container for the information about pre-computed nearest neighbours per class."""

    neighbour_dist_dict_per_class: Dict[int, Dict[int, np.ndarray]]
    neighbour_ind_dict_per_class: Dict[int, Dict[int, np.ndarray]]
    neighbour_error_dict_per_class: Dict[int, Dict[int, np.ndarray]]


class ModelWithConfidence:
    """This class creates a model which returns a prediction and a confidence interval."""

    def __init__(
            self,
            point_pred_model: _ClassificationPointPredictionModel,
            x_train: np.ndarray,
            y_train: Iterable[int],
            macest_model_params: MacestConfModelParams = MacestConfModelParams(),
            precomputed_neighbour_info: Optional[PrecomputedNeighbourInfo] = None,
            graph: Optional[Dict[int, nmslib.dist.FloatIndex]] = None,
            search_method_args: HnswGraphArgs = HnswGraphArgs(),
            training_preds_by_class: Optional[Dict[int, np.ndarray]] = None,
            verbose_training: bool = True,
            empirical_conflict_constant: float = 0.5,
    ):
        """
        Init.

        :param point_pred_model: A sklearn like model which takes some variables x and outputs a
            point prediction y
        :param x_train: The X variables used to train the model
        :param y_train: The target variables used to train the model
        :param _num_neighbours: parameter; the number of neighbours definining local neighbourhood
        :param _alpha: parameter; The linear scaling for the distance function
        :param _beta: parameter; The polynomial scaling for the distance function
        :param precomputed_distance_to_neighbours: Pre-computed Nearest neighbour distances for
            calibration data, used to speed up training
        :param precomputed_index_of_neighbours: Pre-computed indices for Nearest neighbours for
            calibration data, used to speed up training
        :param graph: The graph used in hnsw nearest neighbour search method
        :param search_method_args: The hyper-parameters passed to nmslib index object to configure
            the search
        :param verbose_training:  If true, information such as training progress will be shown
        :param empirical_conflict_constant: Constant to set confidence conflicting predictions
            calculated during calibration
        """
        self.point_pred_model = point_pred_model
        self.x_train = x_train
        self.y_train = y_train
        self.macest_model_params = macest_model_params
        self._num_neighbours = macest_model_params.num_neighbours
        self._alpha = macest_model_params.alpha
        self._beta = macest_model_params.beta
        self._temp = macest_model_params.temp
        self.graph = graph
        self.search_method_args = search_method_args
        self._check_consistent_search_method_args()
        self._check_data_consistent_with_search_args()

        self.training_preds_by_class = training_preds_by_class
        if training_preds_by_class is None:
            self.training_preds_by_class = {
                key: self.predict(self.x_train[y_train == key])
                for key in np.unique(self.y_train)
            }

        self.precomputed_neighbour_info = precomputed_neighbour_info

        if not self.precomputed_neighbour_info:
            self.distance_to_neighbours = None
            self.index_of_neighbours = None
            self.error_on_neighbours = None
        else:
            self.distance_to_neighbours = (
                self.precomputed_neighbour_info.neighbour_dist_dict_per_class
            )
            self.index_of_neighbours = (
                self.precomputed_neighbour_info.neighbour_ind_dict_per_class
            )
            self.error_on_neighbours = (
                self.precomputed_neighbour_info.neighbour_error_dict_per_class
            )

        self.search_method_args = search_method_args

        self._nclasses = np.unique(self.y_train)
        self.point_preds = None
        self.verbose_training = verbose_training
        self.empirical_conflict_constant = empirical_conflict_constant

    def predict(self, x_star: np.ndarray) -> np.ndarray:
        """
        Compute the model point prediction for a given points(s) x_star.

        :param x_star: The point(s) at which we want to predict
        :return: A point prediction for the given x_star
        """
        return self.point_pred_model.predict(x_star)

    def build_class_graphs(self) -> Dict[int, nmslib.dist.FloatIndex]:
        """
        Build a HNSW graph per class.

        :return: A dictionary containing a queryable HNSW graph for each class
        """
        if self.graph is None:
            prec_graph = {}
            graph_init_args = self.search_method_args.init_args
            graph_construction_args = self.search_method_args.construction_args
            graph_query_args = self.search_method_args.query_args
            iterator = np.unique(self.y_train)

            for cls in iterator:
                class_cond = self.y_train == cls
                index = nmslib.init(**graph_init_args)
                index.addDataPointBatch(self.x_train[class_cond])
                index.createIndex(graph_construction_args)
                index.setQueryTimeParams(graph_query_args)
                prec_graph[cls] = index
            self.graph = prec_graph
        return self.graph

    def calc_dist_to_neighbours(
            self, x_star: np.ndarray, cls: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the distance to nearest neighbours, the index of them within the class graph and
        the number of errors on those k neighbours.

        :param x_star: The point(s) we want to find the neighbours for
        :param cls: The target class, we ony want to find neighbours from one class at a time

        :return: The distance to k nearest neighbours and the indices of the k closest neighbours
            in the class graph
        """
        if (self.distance_to_neighbours and self.index_of_neighbours) is None:
            if self.graph is None:
                self.build_class_graphs()
            neighbours = np.array(
                self.graph[cls].knnQueryBatch(  # type: ignore
                    x_star, k=self._num_neighbours, num_threads=num_threads_available
                )
            )
            class_dist = neighbours[:, 1, :].clip(min=10 ** -15)
            class_ind = neighbours[:, 0, :].astype(int)
            if self.training_preds_by_class is None:
                raise ValueError("training_preds_by_class has already been cached")
            class_preds = self.training_preds_by_class[cls]
            class_error = np.array(
                [class_preds[class_ind[j]] != cls for j in range(x_star.shape[0])]
            )
        else:
            if self.distance_to_neighbours is None:
                raise ValueError(
                    "distance_to_neighbours has not been cached during training"
                )
            if self.index_of_neighbours is None:
                raise ValueError(
                    "index_to_neighbours has not been cached during training"
                )
            if self.error_on_neighbours is None:
                raise ValueError(
                    "error_on_neighbours has not been cached during training"
                )

            class_dist = self.distance_to_neighbours[cls][self._num_neighbours]
            class_ind = self.index_of_neighbours[cls][self._num_neighbours]
            class_error = self.error_on_neighbours[cls][self._num_neighbours]

        neighbour_info = NearestNeighbourInClassInfo(
            class_dist=class_dist, class_ind=class_ind, class_error=class_error
        )
        return neighbour_info

    def calc_linear_distance_error_func(
            self, local_distance: np.ndarray, local_error: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the parametric linear distance function using the local error and distance.

        :param local_distance:  The distance to the k nearest neighbours
        :param local_error: list of booleans representing the error rate
        :return: A weighted sum of average error and average distance
        """
        dist = self._alpha * np.average(
            local_distance.clip(min=10 ** -15),
            weights=np.arange(local_distance.shape[1], 0, -1),
            axis=1,
        )

        error = self._beta * np.average(
            local_error, weights=1.0 / (1 + local_distance), axis=1
        )
        return dist, error

    def predict_proba(
            self, x_star: np.ndarray, change_conflicts: bool = False,
    ) -> np.ndarray:
        """
        Compute a confidence score for each class for a given points(s) x_star.

        :param x_star: The point to predict confidently
        :param change_conflicts: Boolean, true means conflicting predictions between macest and
            point prediction are set to an empirical constant.

        :return: A confidence score for each class
        """
        classes = self._nclasses
        av_dist_func = np.zeros(shape=(len(classes), x_star.shape[0]))
        for i, cls in enumerate(classes):
            class_dist, _, class_error = self.calc_dist_to_neighbours(x_star, cls)

            dist, error = self.calc_linear_distance_error_func(class_dist, class_error)
            av_dist_func[i, :] = dist.clip(min=10 ** -10) + error
        relative_conf = self._calc_relative_distance_softmax_normalisation(av_dist_func)
        if change_conflicts:
            relative_conf = self._renormalise_conf_with_empirical_constant(
                x_star, relative_conf
            )
        return relative_conf

    def predict_confidence_of_point_prediction(
            self, x_star: np.ndarray, change_conflicts: bool = False,
    ) -> np.ndarray:
        """
        Estimate a single confidence score, this represents the confidence of the point prediction
        being correct rather than a confidence score for each class.

        :param x_star: The point to predict confidently
        :param change_conflicts: Boolean, true means conflicting predictions between macest and
            point prediction are set to an empirical constant

        :return: The confidence in the point prediction being correct

        """
        if self.point_preds is not None:
            point_prediction = self.point_preds
        else:
            point_prediction = self.predict(x_star)

        class_confidence = self.predict_proba(x_star, change_conflicts)

        point_prediction_confidence = class_confidence[
            np.arange(len(class_confidence)), point_prediction
        ].clip(max=1 - 10 ** -15)
        return point_prediction_confidence

    def _calc_relative_distance_softmax_normalisation(
            self, average_distance_error_func: np.ndarray,
    ) -> np.ndarray:
        """
        Take a vector of distance functions, we then scale these by the mean distance across
        classes so that we have the relative distances between then, this vector is then normalised
        these via temperature scaled softmax.

        :param average_distance_error_func:
        :return: The confidence estimates for each class after normalising via softmax

        """
        average_distance_error_func = (
                self._temp
                * average_distance_error_func
                / np.median(average_distance_error_func, axis=0)
        )
        relative_conf = softmax(-average_distance_error_func.T, axis=1)
        return relative_conf

    def _renormalise_conf_with_empirical_constant(
            self, x_star: np.ndarray, conf_array: np.ndarray
    ) -> np.ndarray:
        """
        Change conflicting predictions to the empirically learnt constant probability learnt during \
        training, by default this feature will be off as it can introduce bias.

        :param x_star:
        :param conf_array:
        :return:
        """
        conflicting_predictions = self.find_conflicting_predictions(x_star)
        point_prediction = self.predict(x_star)
        for idx in range(x_star.shape[0]):
            if idx in conflicting_predictions:
                conf_array[idx, point_prediction[idx]] = 0.0
                conf_array[idx] = conf_array[idx] / conf_array[idx].sum()
                conf_array[idx] = conf_array[idx] * (
                        1 - self.empirical_conflict_constant
                )
                conf_array[
                    idx, point_prediction[idx]
                ] = self.empirical_conflict_constant + np.random.normal(0, 0.01)

        return conf_array

    def find_conflicting_predictions(self, x_star: np.ndarray) -> np.ndarray:
        """
        Find predictions where max confidence according to macest is different to the point
        prediction.

        :param x_star:
        :return:
        """
        class_confidence = self.predict_proba(x_star, )
        point_prediction = self.predict(x_star, )
        max_confidence_prediction = np.argmax(class_confidence, axis=1)
        conflicting_predictions = np.argwhere(
            max_confidence_prediction != point_prediction
        ).flatten()
        return conflicting_predictions

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
        init_args = self.search_method_args.init_args

        if 'space' not in list(init_args.keys()):
            raise ValueError('You must pass a space in your search method init args')

        index = nmslib.init(**init_args)

        space = init_args['space']
        if 'sparse' in space:
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
        init_args = self.search_method_args.init_args

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
    """Class which provides methods used when fitting MACEst model."""

    def __init__(
            self,
            init_conf_model: ModelWithConfidence,
            x_cal: np.ndarray,
            y_cal: np.ndarray,
            param_range: SearchBounds = SearchBounds(),
    ):
        """
        Init.

        :param init_conf_model: an initialised ModelWithConfidence object that we want to fit
        :param x_cal: The X variables that we will use to calibrate the confidence predictions
        :param y_cal: The target variables that we will use to calibrate the confidence predictions
        :param param_range: The bounds on the MACEst parameter space we want to search
        """
        self.model = init_conf_model
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.param_range = param_range
        self._precompute_graph()
        self.precomputed_neighbours = self._precompute_neighbours
        self.precomputed_distance = self.precomputed_neighbours[0]
        self.precomputed_index = self.precomputed_neighbours[1]
        self.precomputed_error = self.precomputed_neighbours[2]
        self._n_classes = len(np.unique(self.model.y_train))
        self.model.point_preds = self.model.predict(self.x_cal)
        self.model.distance_to_neighbours = self.precomputed_distance
        self.model.index_of_neighbours = self.precomputed_index
        self.model.error_on_neighbours = self.precomputed_error

    def _precompute_graph(self) -> None:
        """Pre-compute the hnsw index graph."""
        self.model.build_class_graphs()

    @property
    def _precompute_neighbours(self) -> PrecomputedNeighbourInfo:
        """
        Pre-compute the nearest neighbours and their distances.

        :return:
        """
        min_nbrs = self.param_range.k_bounds[0]
        max_nbrs = self.param_range.k_bounds[1]
        num_nbrs = np.arange(min_nbrs, max_nbrs + 0.1, 1)
        cls_dist_dict: Dict[int, Dict[int, np.ndarray]] = {}
        cls_ind_dict: Dict[int, Dict[int, np.ndarray]] = {}
        cls_error_dict: Dict[int, Dict[int, np.ndarray]] = {}

        iterator = np.unique(self.model.y_train)

        x_cal_len_array = np.arange(self.x_cal.shape[0])

        for _class_num in iterator:
            class_num = int(_class_num)
            dist_dict: Dict[int, np.ndarray] = {}
            ind_dict: Dict[int, np.ndarray] = {}
            error_dict: Dict[int, np.ndarray] = {}

            max_neighbours = np.array(
                self.model.graph[class_num].knnQueryBatch(  # type: ignore
                    self.x_cal, k=max_nbrs, num_threads=num_threads_available
                )
            )
            max_dist = max_neighbours[x_cal_len_array, 1]
            max_ind = max_neighbours[x_cal_len_array, 0]
            for k in num_nbrs:
                dist = max_dist[x_cal_len_array, 0: int(k)]
                ind = max_ind[x_cal_len_array, 0: int(k)]
                cls_preds = self.model.training_preds_by_class[class_num]  # type: ignore
                error = np.array(
                    [
                        cls_preds[ind[j].astype(int)] != class_num
                        for j in range(self.x_cal.shape[0])
                    ]
                )  # type: ignore

                dist_dict[k] = dist
                ind_dict[k] = ind
                error_dict[k] = error

            cls_dist_dict[class_num] = dist_dict
            cls_ind_dict[class_num] = ind_dict
            cls_error_dict[class_num] = error_dict

        pre_computed_neighbour_info = PrecomputedNeighbourInfo(
            neighbour_dist_dict_per_class=cls_dist_dict,
            neighbour_ind_dict_per_class=cls_ind_dict,
            neighbour_error_dict_per_class=cls_error_dict,
        )
        return pre_computed_neighbour_info

    def set_macest_model_params(self) -> MacestConfModelParams:
        """
        Assign the MACEst model parameters.

        :return: _alpha, _beta, k , Temp
        """
        params = MacestConfModelParams(
            alpha=self.model._alpha,
            beta=self.model._beta,
            num_neighbours=self.model._num_neighbours,
            temp=self.model._temp,
        )
        self.model.macest_model_params = params
        return params

    def loss(self, params: MacestConfModelParams) -> float:
        """
        Return the loss for a given set of MACEst parameters, this will be optimised to find
            optimal parameters.

        :param params: A tuple containing the model parameters
        :return: The ece loss function for the model
        """
        (
            self.model._alpha,
            self.model._beta,
            self.model._num_neighbours,
            self.model._temp,
        ) = params

        self.model._num_neighbours = int(np.round(self.model._num_neighbours))

        pred_conf = self.model.predict_confidence_of_point_prediction(self.x_cal)
        return expected_calibration_error(self.model.point_preds, self.y_cal, pred_conf)

    def fit(
            self,
            optimiser: Literal["de"] = "de",
            optimiser_args: Optional[Dict[Any, Any]] = None,
    ) -> ModelWithConfidence:
        """
        Fit MACEst model using the calibration data.

        :param optimiser: The optimisation method
        :param optimiser_args: Any arguments for the optimisation strategy

        :return: A ModelWithConfidence object with the parameters that minimises the loss function
        """
        if optimiser_args is None:
            optimiser_args = {}
        alpha_bounds = self.param_range.alpha_bounds
        beta_bounds = self.param_range.beta_bounds
        k_bounds = self.param_range.k_bounds
        temperature_bounds = self.param_range.temperature_bounds

        bounds = (alpha_bounds, beta_bounds, k_bounds, temperature_bounds)

        if optimiser == "de":
            result = differential_evolution(self.loss, bounds=bounds, **optimiser_args)

        else:
            raise ValueError(
                f" optimiser: {optimiser} is not implemented, currently the only optimisation strategy "
                f"implemented is differential evolution "
            )

        log.info(f" min_loss = {result.fun}")

        alpha, beta, k, temp = result.x

        log.info(f" best_alpha: {alpha}")
        log.info(f" best_beta: {beta}")
        log.info(f" best_k: {k}")
        log.info(f"best_Temp: {temp}")

        self.model._alpha = alpha
        self.model._beta = beta
        self.model._num_neighbours = int(np.round(k, 0))
        self.model._temp = temp

        self.model.macest_model_params = self.set_macest_model_params()

        point_preds = self.model.predict(self.x_cal)
        conflicts = self.model.find_conflicting_predictions(self.x_cal)
        self.model.empirical_conflict_constant = np.array(
            point_preds[conflicts] == self.y_cal[conflicts]
        ).mean()

        self.model.distance_to_neighbours = None
        self.model.index_of_neighbours = None
        self.model.error_on_neighbours = None
        self.model.point_preds = None

        return self.model
