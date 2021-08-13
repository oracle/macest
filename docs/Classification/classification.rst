Classification
**************************

Brief Outline of Algorithm
==============================
For any labelled dataset (x,y), first split the dataset into four groups:

1. Point prediction model training data
2. HNSW graph data
3. MACEst parameters calibration data
4. Unseen Test data

First train a point prediction algorithm (e.g. SVM, Random Forest, etc).

For each data point find k nearest neighbours within each class from the Graph data.

Calculate the average distance (epistemic uncertainty) to and error rate (aleatoric uncertainty) of these k neighbours,

Define simple linear sum of these two terms.

This sum is then normalised (via softmax) to produce confidence scores

The co-efficients of this linear sum are learnt during the calibration phase by minimising a calibration loss function.

Examples
=====================
.. toctree::
   :maxdepth: 1

   examples

Classification API
===================

.. toctree::
   :maxdepth: 1

   classification_models
   classification_utils
   classification_plots
   classification_metrics






