# MACEst (Model Agnostic Confidence Estimator)
## What is MACEst?
MACEst is a confidence estimator that can be used alongside any model (regression or 
classification) which uses previously seen data (i.e. any supervised learning model) to produce a 
point prediction.

In the regression case, MACEst produces a _confidence interval_ about the point prediction, e.g. 
"the point prediction is 10 and I am 90% confident that the prediction lies between 8 and 12."

In Classification MACEst produces a _confidence score_ for the point prediction. e.g. the point 
prediction is class 0 and I am 90% sure that the prediction is correct.

MACEst produces well-calibrated confidence estimates, i.e. 90% confidence means that you will on 
average be correct 90% of the time. 
It is also aware of the model limitations i.e. when a model is being asked to predict a point which 
it does not have the necessary knowledge (data) to predict confidently. 
In these cases MACEst is able to incorporate the (epistemic) uncertainty due to this and return a 
very low confidence prediction (in regression this means a large prediction interval).

## Why use MACEst ?
Machine learning has become an integral part of many of the tools that are used every day. 
There has been a huge amount of progress on improving the global accuracy of machine learning 
models but calculating how likely a single prediction is to be correct has seen considerably less 
progress.

Most algorithms will still produce a prediction, even if this is in a part of the feature space the 
algorithm has no information about. 
This could be because the feature vector is unlike anything seen during training, or because the 
feature vector falls in a part of the feature space where there is a large amount of uncertainty 
such as if the border between two classes overlaps.
In cases like this the prediction may well be meaningless. 
In most models, it is impossible to distinguish this sort of meaningless prediction from a sensible 
prediction. 
MACEst addresses this situation by providing an additional confidence estimate.

In some areas such as Finance, Infrastructure, or Healthcare, making a single bad prediction can 
have major consequences.
It is important in these situations that a model is able to understand how likely any prediction it 
makes is to be correct before acting upon it. 
It is often even more important in these situations that any model *knows what it doesn't know* so 
that it will not blindly make bad predictions.

## Summary of the Methodology
### TL;DR
MACEst produces confidence estimates for a given point x by considering two factors:
1. How accurate is the model when predicting previously seen points that are **similar** to x? 
Less confident if the model is less accurate in the region close to x.
2. How **similar** is x to the points that we have seen previously? 
Less confident if x is not **similar** to the data used to train the model.

### Longer Explanation
MACEst seeks to provide reliable confidence estimates for both regression and classification. 
It draws from ideas present in trust scores, conformal learning, Gaussian processes, and Bayesian 
modelling.

The general idea is that confidence is a local quantity. 
Even when the model is accurate globally, there are likely still some predictions about which it 
should not be very confident. 
Similarly, if the model is not accurate globally, there may still be some predictions for which the 
model can be very confident about.

To model this local confidence for a given prediction on a point x, we define the local 
neighbourhood by finding the k nearest neighbours to x. 
We then attempt to directly model the two causes of uncertainty, these are:
1. _Aleatoric Uncertainty_: Even with lots of (possibly infinite) data there will be some 
variance/noise in the predictions.
Our local approximation to this will be to define a local accuracy estimate. i.e. for the k nearest 
neighbours how accurate were the predictions?
2. _Epistemic Uncertainty_: The model can only know relationships learnt from the training data. 
If the model has not seen any data point similar to x then it does not have as much knowledge about 
points like x, therefore the confidence estimate should be lower. 
MACEst estimates this by calculating how **similar** x is to the k nearest (most similar) points 
that it has previously seen.

We define a simple parametric function of these two quantities and calibrate this function so that 
our confidence estimates approximate the empirical accuracy, i.e. 90% confident -> 90% correct on 
average. 
By directly modelling these two effects, MACEst estimates are able to encapsulate the local 
variance accurately whilst also being aware of when the model is being asked to predict a point 
that is very different to what it has been trained on. 
This will make it robust to problems such as overconfident extrapolations and out of sample 
predictions.

### Example
If a model has been trained to classify images of cats and dogs, and we want to predict an image of 
a poodle, we find the k most poodle-like cats and the k most poodle-like dogs. 
We then calculate how accurate the model was on these sets of images, and how similar the poodle is 
to each of these k cats and k dogs. We combine these two to produce a confidence estimate for each 
class.

As the poodle-like cats will likely be strange cats, they will be harder to classify and the 
accuracy will be lower for these than the poodle-like dogs this combined with the fact that image 
will be considerably more similar to poodle-like dogs the confidence of the dog prediction will be 
high.

If we now try to classify an image of a horse, we find that the new image is very **dissimilar** to 
both cats and dogs, so the similarity term dominates and the model will return an approximately 
uniform distribution, this can be interpreted as MACEst saying "I don't know what this is because 
I've never seen an image of a horse!".

## Getting Started
To install MACEst run the following cmd:
```shell script
pip install macest
```

Or add `macest` to your project's `requirements.txt` file as a dependency. 

### Software Prerequisites
To import and use MACEst we recommend Python version >= `3.6.8`. 

## Basic Usage
Below shows examples of using MACEst for classification and regression.
For more examples, and advanced usage, please see the example [notebooks](./notebooks).

### Classification 
To use MACEst for a classification task, the following example can be used:
``` python

   import numpy as np
   from macest.classification import models as cl_mod
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
``` 

### Regression
To use MACEst for a regression task, the following example can be used:
``` python
   import numpy as np
   from macest.regression import models as reg_mod
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
 ```

### MACEst with sparse data (see notebooks for more details)
```python
import scipy
from scipy.sparse import csr_matrix 
from scipy.sparse import random as sp_rand
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from macest.classification import models as clmod
import nmslib 

n_rows = 10**3
n_cols = 5 * 10**3
X = csr_matrix(sp_rand(n_rows, n_cols))
y = np.random.randint(0, 2, n_rows)

X_pp_train, X_conf_train, y_pp_train, y_conf_train = train_test_split(X, y, test_size=0.66, random_state=10)
X_conf_train, X_cal, y_conf_train, y_cal = train_test_split(X_conf_train, y_conf_train,
                                                            test_size=0.5, random_state=0)
X_cal, X_test, y_cal,  y_test, = train_test_split(X_cal, y_cal, test_size=0.5, random_state=0)

model = RandomForestClassifier(random_state=0,
                               n_estimators=800,
                               n_jobs=-1)

model.fit(csr_matrix(X_pp_train), y_pp_train)

param_bounds = clmod.SearchBounds(alpha_bounds=(0, 500), k_bounds=(5, 15))
neighbour_search_params = clmod.HnswGraphArgs(query_args=dict(ef=1100),
                                              init_args=dict(method="hnsw",
                                                             space="cosinesimil_sparse",
                                                             data_type=nmslib.DataType.SPARSE_VECTOR))
macest_model = clmod.ModelWithConfidence(model,
                                       X_conf_train,
                                       y_conf_train,
                                       search_method_args=neighbour_search_params)

macest_model.fit(X_cal, y_cal)

macest_point_prediction_conf = macest_model.predict_confidence_of_point_prediction(X_test)

```

## Contributing
See the [`CONTRIBUTING.md`](./CONTRIBUTING.md) file for information about contributing to MACEst.


## License
Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.

This library is licensed under Universal Permissive License (UPL) 1.0 as shown at 
https://oss.oracle.com/licenses/upl

See [LICENSE.txt](./LICENSE.txt) for more details.
