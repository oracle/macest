# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
import numpy as np

from macest.model_selection import KFoldConfidenceSplit


def test_kfold_init():
    kfold = KFoldConfidenceSplit()

    np.testing.assert_(10, kfold.n_splits)
    np.testing.assert_(True, kfold.shuffle)
    np.testing.assert_array_equal((0.5, 0.3, 0.2), kfold.pp_train_graph_cal_split)


def test_kfold_splits():
    kfold = KFoldConfidenceSplit(
        n_splits=2, shuffle=False, pp_train_graph_cal_split=(0.6, 0.2, 0.2)
    )

    # n_splits means len(x_test = 0.5 * len(X))

    x = np.arange(10).reshape(-1, 1)

    for train_idxs, conf_cal_idxs, conf_graph_idxs, test_idxs in kfold.split(x):
        # test lengths are correct for split
        assert len(test_idxs) == 5
        assert len(train_idxs) == 3
        assert len(conf_graph_idxs) == 1
        assert len(conf_cal_idxs) == 1

        # test that sets are mututally exclusive
        assert any(
            idx not in np.concatenate((train_idxs, conf_cal_idxs, conf_graph_idxs))
            for idx in test_idxs
        )
        assert any(
            idx not in np.concatenate((test_idxs, conf_cal_idxs, conf_graph_idxs))
            for idx in train_idxs
        )
        assert any(
            idx not in np.concatenate((test_idxs, conf_cal_idxs, train_idxs))
            for idx in conf_graph_idxs
        )
        assert any(
            idx not in np.concatenate((test_idxs, train_idxs, conf_graph_idxs))
            for idx in conf_cal_idxs
        )

    # change the split ratios

    kfold = KFoldConfidenceSplit(
        n_splits=2, shuffle=False, pp_train_graph_cal_split=(0.8, 0.1, 0.1)
    )

    # n_splits means len(x_test = 0.5 * len(X))

    x = np.arange(20).reshape(-1, 1)

    for train_idxs, conf_cal_idxs, conf_graph_idxs, test_idxs in kfold.split(x):
        # test lengths are correct for split
        assert len(test_idxs) == 10
        assert len(train_idxs) == 8
        assert len(conf_graph_idxs) == 1
        assert len(conf_cal_idxs) == 1

        # test that sets are mututally exclusive
        assert any(
            idx not in np.concatenate((train_idxs, conf_cal_idxs, conf_graph_idxs))
            for idx in test_idxs
        )
        assert any(
            idx not in np.concatenate((test_idxs, conf_cal_idxs, conf_graph_idxs))
            for idx in train_idxs
        )
        assert any(
            idx not in np.concatenate((test_idxs, conf_cal_idxs, train_idxs))
            for idx in conf_graph_idxs
        )
        assert any(
            idx not in np.concatenate((test_idxs, train_idxs, conf_graph_idxs))
            for idx in conf_cal_idxs
        )
