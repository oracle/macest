# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
"""Module for recreating K-fold with an extra set needed for MACEst."""
from typing import Optional, Sequence, Iterator, Tuple
import numpy as np


class KFoldConfidenceSplit:
    """Equivalent api to sklearn.model_selection.KFold for Confidence calibration splits."""

    def __init__(self,
                 n_splits: int = 10,
                 shuffle: bool = True,
                 random_state: Optional[int] = None,
                 pp_train_graph_cal_split: Sequence[float] = (0.5, 0.3, 0.2),
                 ):
        """
        The constructor for KFoldConfidenceSplit. 

        :param n_splits: Number of folds, Must be at least 2 
        :param shuffle: Whether to shuffle the data before splitting into batches
        :param random_state: If int, random_state is the seed used by the random number generator
        :param pp_train_graph_cal_split: The fraction of training data to be used when splitting 
            between the data used to train the point prediction model, the data to build the hnsw 
            graph and the MACEst model calibration parameters
        """  # noqa
        self.n_splits = n_splits
        if self.n_splits < 2:
            raise ValueError('number of splits must be at least 2')
        self.shuffle = shuffle
        self.random_state = random_state
        self.pp_train_graph_cal_split = pp_train_graph_cal_split
        if abs(np.array(self.pp_train_graph_cal_split).sum() - 1.0) > 10 ** -6:
            raise ValueError("split of training data must sum to 1")

    def split(self, data: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split the data into 4 distinct sets for k folds.

        These are:
        1. Point prediction model training data
        2. HNSW graph data
        3. MACEst parameters calibration data
        4. Unseen Test data

        :param data: Training data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        test_bound = int(1 / self.n_splits * data.shape[
            0])

        non_test_data = 1 - (1.0 / self.n_splits)

        point_prediction_train_size = self.pp_train_graph_cal_split[0] * non_test_data
        confidence_calibration_size = self.pp_train_graph_cal_split[1] * non_test_data

        train_bound = int(test_bound + point_prediction_train_size * len(data))
        conf_cal_bound = int(train_bound + confidence_calibration_size * len(data))

        possible_idxs = np.arange(len(data))

        if self.shuffle:
            if self.random_state:
                np.random.seed(self.random_state)
            np.random.shuffle(possible_idxs)

        for fold in range(self.n_splits):
            test_idx = possible_idxs[0:test_bound]
            train_idx = possible_idxs[test_bound:train_bound]
            conf_cal_idx = possible_idxs[train_bound:conf_cal_bound]
            conf_graph_idx = possible_idxs[conf_cal_bound:]

            possible_idxs = np.roll(possible_idxs, -test_bound)  # This is cool!

            yield train_idx, conf_cal_idx, conf_graph_idx, test_idx
