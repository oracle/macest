MACEst (Model Agnostic Confidence Estimator)
*******************************************

What is MACEst ?
===================
MACEst is a confidence estimator that can be used alongside any model (regression or classification) which uses
previously seen data (i.e. any supervised learning model) to produce a point prediction.

In the regression case, MACEst produces a *confidence interval* about the point prediction, i.e. the point prediction is
10 and I am 90% confident that the prediction lies between 8 and 12.

In Classification MACEst produces a *confidence score* for the point prediction. i.e. the point prediction is class 0 and
I am 90% sure that the prediction is correct.

MACEst produces well calibrated confidence estimates, i.e. 90% confidence means that you will on average be correct 90%
of the time. It is also aware of the model limitations i.e. when a model is being asked to predict a point which it does
not have the necessary knowledge to predict confidently. In these cases MACE is able to return a proxy for I don't know.

Why use MACEst ?
===================
Machine learning has become an integral part of many of the tools that are used every day, there has been a huge amount
of progress on improving the global accuracy of machine learning models, calculating how likely a single prediction is to be
correct has seen considerably less progress.

In some areas such as Infrastructure or Healthcare making a single bad predictions can have major consequences.
It is important in these situations that a model is able to understand how likely any prediction it makes is to be
correct before acting upon it. It is often even more important in these situations that any model
*knows what it doesn't know* so that it will not blindly make bad predictions.

How to use MACEst ?
===================
MACEst provides a simple API which can be easily incorporated standard point-prediction pipelines with minimal
changes and a short calibration stage after the point prediction model has been trained.

See the getting started and examples below for more information

.. toctree::
   :maxdepth: 1

   install
   getting_started
   Classification/classification
   Regression/regression
   model_selection



