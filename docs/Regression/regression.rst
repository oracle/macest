Regression
**************************

Brier Outline of Algorithm
==============================
For any labelled dataset (x,y), first split the dataset into four groups:

1. Point prediction model training data
2. HNSW graph data
3. MACEst parameters calibration data
4. Unseen Test data

First train a regression point prediction algorithm (e.g. SVM, Random Forest, ect).

Then on the calibration data, look at the distribution of errors, i.e. is it roughly gaussian? laplacian ?
(Can perform goodness of fit tests to check this)

First train a point prediction algorithm (e.g. SVM, Random Forest, ect).

For each data point find k nearest neighbours within each class from the Graph data.

Calculate the average distance (epistemic uncertainty) to and error rate (aleatoric uncertainty) of these k neighbours,
Define simple linear sum of these two terms.

The sum of these then becomes the width scale parameter for the distribution, e.g. the standard deviation for a gaussian.

The co-efficients of this linear sum are learnt during the calibration phase by minimising a calibration loss function.



Examples
===================
.. toctree::
   :maxdepth: 1

   examples

Regression API
===================

.. toctree::
   :maxdepth: 1

   regression_models
   regression_metrics
   regression_plots
