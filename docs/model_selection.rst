Validating Confidence estimators
************************************

Summary
================================
Kfold validation is a very popular way of comparing prediction metrics in machine learning. Kfold generally splits the
data into k splits Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
As MACE requires an additional two splits, we provide this utility function. It works on the same principle and api
as the standard KFold function in sklearn but just further splits the training data into 3 splits. This means as well
as the unseen validation fold, for each k -1 training folds we have 3 splits:

1. Point prediction model training data
2. HNSW graph data
3. MACEst parameters calibration data


Basic Usage
===============
.. code-block:: python

    import numpy as np
    from macest.model_selection import KFoldConfidenceSplit

    kfold = KFoldConfidenceSplit(n_splits =5, shuffle = True)
    X = np.arange(100)
    for train_idxs, conf_cal_idxs, conf_graph_idxs, test_idxs in kfold.split(X):
        X_pp_train = X[train_idxs]
        y_pp_train =  y[train_idxs]

        X_test = X[test_idxs]
        y_test = y[test_idxs]

        X_cal = X[conf_cal_idxs]
        y_cal = y[conf_cal_idxs]

        X_graph = X[conf_graph_idxs]
        y_graph = y[conf_graph_idxs]


Kfold for Confidence estimation
===================================
.. automodule:: macest.model_selection
   :members:
