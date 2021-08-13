Getting Started
*********************************


Summary of method
=============================
MACEst seeks to provide reliable confidence estimates for both regression and classification, it draws from ideas present
in trust scores, conformal learning, Gaussian process' and Bayesian modelling. (add refs for these) (yes, add refs)

The general idea is that confidence is a local quantity, i.e. if the model is terrible globally then there are still
some predictions for which the model can be very confident about, similarly if the model is very accurate
globally then there are still some predictions which should not be very confident.

To model this local confidence for a given prediction on a point x, we define the local neighbourhood by finding the k
nearestneighbours to x. We then attempt to directly model the two causes of uncertainty, these are:

1. Aleatoric Uncertainty: Even with lots of (even infinite) data there will be some variance/noise in the predictions.
Our local approximation to this will be to define a local accuracy estimate. i.e. for the k nearest neighbours how
accurate were the predictions ?

2. Epistemic Uncertainty: The model can only know relationships learnt from the training data so if the model has not
seen any data point similar to x then it does not have as much knowledge about points like x therefore the confidence
estimate should be lower. MACE estimates this by calculating how **similar** x is to the k nearest (most similar) points
that it has previously seen.

We can then define a simple parametric function of these two quantities and calibrate this function so that our
confidence estimates approximate the empirical accuracy i.e. 90% confident -> 90% correct on average. By modelling these
two effects directly MACE estimates are able to encapsulate the local variance accurately whilst also being aware of when
the model is being asked to predict a point that is very different to what it's been trained on. This will make it robust
to problems such as overconfident extrapolations and out of sample predictions.

TL;DR
---------------------------------
MACEst produces confidence estimates for a given point x by considering two factors;

1. How accurate is the model when predicting previously seen points that are **similar** to x ? Less confident is the
model if less accurate in the region close to x.

2. How **similar** is x to the points that we've seen previously ? Less confident if x is not **similar** to the data used to
train the model.

Example
--------------------------
If a model has been trained to classify images of cats and dogs, and we want to predict an image of a poodle, we find
the k most poodle-like cats and the k most poodle-like dogs. We then calculate how accurate the model was on these sets
of images, and how similar the poodle is to each of these k cats and k dogs. We combine these two to produce a
confidence estimate for each class.

As the poodle-like cats will likely be strange cats, they will be harder to classify and the accuracy will be lower for
these than the poodle-like dogs this combined with the fact that image will be considerably more similar to poodle-like
dogs the confidence of the dog prediction will be high.

If we now try to classify an image of a horse, we find that the new image is very **dissimilar** to both cats and dogs,
so the similarity term dominates and the model will return an approximately uniform distribution, this can be
interpreted as MACE saying I don't know what this is because I've never seen an image of a horse!


Basic Usage
===============

Classification
---------------

.. code-block:: python

   import numpy as np
   from macest.classification import classification_models as cl_mod
   from sklearn.ensemble import RandomForestClassifier
   from sklearn import datasets
   from sklearn.model_selection import train_test_split

   X,y = datasets.make_circles(n_samples= 2 * 10**4, noise = 0.4, factor =0.001)

   X_pp_train, X_conf_train, y_pp_train, y_conf_train  = train_test_split(X,
                                                                          y,
                                                                          test_size=0.66,
                                                                          random_state=10)

   X_conf_train, X_cal, y_conf_train, y_cal = train_test_split(X_conf_train,
                                                               y_conf_train,
                                                               test_size=0.5,
                                                               random_state=0)

   X_cal, X_test, y_cal,  y_test, = train_test_split(X_cal,
                                                     y_cal,
                                                     test_size=0.5,
                                                     random_state=0)

   point_pred_model = RandomForestClassifier(random_state =0,
                                             n_estimators =800,
                                             n_jobs =-1)

   point_pred_model.fit(X_pp_train,
                        y_pp_train)

   macest_model = cl_mod.ModelWithConfidence(point_pred_model,
                                          X_conf_train,
                                          y_conf_train)

   macest_model.fit(X_cal, y_cal)

   conf_preds = macest_model.predict_confidence_of_point_prediction(X_test)

Regression
----------------

.. code-block:: python

   import numpy as np
   from macest.regression import regression_models as reg_mod
   from sklearn.linear_model import LinearRegression
   from sklearn.model_selection import train_test_split

   X = np.linspace(0,1,10**3)
   y = np.zeros(10**3)
   y = 2*X*np.sin(2 *X)**2 + np.random.normal(0 , 1 , len(X))

   X_pp_train, X_conf_train, y_pp_train, y_conf_train  = train_test_split(X,
                                                                          y,
                                                                          test_size=0.66,
                                                                          random_state=0)

   X_conf_train, X_cal, y_conf_train, y_cal = train_test_split(X_conf_train, y_conf_train,
                                                            test_size=0.5, random_state=1)

   X_cal, X_test, y_cal,  y_test, =  train_test_split(X_cal,
                                                      y_cal,
                                                      test_size=0.5,
                                                      random_state=1)

   point_pred_model = LinearRegression()
   point_pred_model.fit(X_pp_train[:,None], y_pp_train)

   preds = point_pred_model.predict(X_conf_train[:,None])
   test_error = abs(preds - y_conf_train)
   y_conf_train_var = np.var(train_error)

   macest_model = reg_mod.ModelWithPredictionInterval(point_pred_model,
                                                    X_conf_train[:,None],
                                                    test_error)

   macest_model.fit(X_cal[:,None], y_cal)
   conf_preds = confidence_model.predict_interval(X_test, conf_level=90)
