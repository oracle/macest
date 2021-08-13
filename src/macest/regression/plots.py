# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
"""Module contains plots used for evaluating prediction interval methods."""

import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, Union, Tuple, NamedTuple
from typing_extensions import Literal
from macest.regression.models import (
    ModelWithPredictionInterval,
    _RegressionPointPredictionModel,
)
from macest.regression.metrics import predictions_in_range

log = logging.getLogger()


class PredictionIntervalsForPlot(NamedTuple):
    """Container for interval plot ranges."""

    lower_95: np.ndarray
    lower_90: np.ndarray
    lower_70: np.ndarray

    upper_95: np.ndarray
    upper_90: np.ndarray
    upper_70: np.ndarray


class SortedPredsAndIntervals(NamedTuple):
    """Container for interval plot ranges."""

    sorted_y_true: np.ndarray
    sorted_pred: np.ndarray
    sorted_intervals: PredictionIntervalsForPlot


def plot_pred_with_conf(
    conf_model: ModelWithPredictionInterval,
    x_star: np.ndarray,
    confidence_level: Union[int, float] = 90,
    y_true: Optional[float] = None,
    save: bool = False,
    save_dir: str = "./distribution.png",
) -> None:
    """
    Plot the distribution about some point prediction.

    :param conf_model: A model which outputs a confidence interval about some point prediction
    :param x_star: The set of target variables we want to predict for
    :param y_true: The true value for the target variables (if known)
    :param confidence_level: The level of confidence that you want to see, i.e. 90%
    :param save: True means plot is saved in save_dir
    :param save_dir: Location to save figure

    :return: None
    """
    pred_dist = conf_model.sample_prediction(x_star, nsamples=10 ** 4)

    point_pred = np.mean(pred_dist)
    plt.figure(figsize=(12, 6))
    sns.histplot(pred_dist.T, stat="probability", label="p(y|x)")
    lower, upper, = conf_model.predict_interval(
        x_star, conf_level=np.array(confidence_level),
    ).T

    plt.axvline(lower, linestyle="--", color="b")
    plt.axvline(
        upper, linestyle="--", color="b", label=f"{confidence_level}% confidence"
    )
    plt.axvline(point_pred, color="r", label="point prediction")
    if y_true:
        plt.axvline(y_true, color="y", label="true")
    plt.legend()
    if save:
        plt.savefig(save_dir)


def plot_calibration(
    conf_model: ModelWithPredictionInterval,
    x_star: np.ndarray,
    y_true: np.ndarray,
    save: bool = False,
    save_dir: str = "./",
) -> None:
    """
    Plot the calibration for a confidence model.

    For a set of confidence intervals plot the fraction of the
    points lie within that confidence interval.

    :param conf_model: A model which outputs a confidence interval about some point prediction
    :param x_star: The set of target variables we want to predict for
    :param y_true: The true value for the target variables
    :param save: True means plot is saved in save_dir
    :param save_dir: Location to save figure

    :return: None
    """
    frac_in_int = []

    check_vals = np.array((10, 30, 50, 70, 90, 95))

    for interval in check_vals:
        log.info(f"{interval}th percentile")
        in_range = predictions_in_range(
            y_true, x_star, conf_model, interval, verbose=True
        )
        frac_in_int.append(in_range)

    plt.figure(figsize=(10, 10))
    plt.grid()
    sl = np.arange(-2, 200.001, 1)
    plt.fill_between(sl, 200, sl, alpha=0.3, color="g")
    plt.fill_between(sl, -1, sl, alpha=0.3, color="r")

    plt.plot(sl, sl, linestyle="--", color="y", linewidth=4.0, label="optimal")

    plt.text(
        30.0,
        64,
        "Underconfident",
        fontsize=25,
        rotation=45,
        horizontalalignment="center",
        verticalalignment="center",
    )

    plt.text(
        64,
        30.0,
        "Overconfident",
        fontsize=25,
        rotation=45,
        horizontalalignment="center",
        verticalalignment="center",
    )

    plt.plot(
        check_vals,
        frac_in_int,
        color="b",
        linewidth=3.0,
        label="distance model calibration",
    )

    plt.xlabel("predicted confidence interval (%)", fontsize=18)
    plt.ylabel("points in interval (%)", fontsize=18)
    plt.xlim(-0.05, 102.02)
    plt.ylim(-0.05, 102.02)
    plt.legend(loc="upper left")
    if save:
        plt.savefig(save_dir)


def plot_true_vs_predicted(
    model: _RegressionPointPredictionModel,
    conf_model: ModelWithPredictionInterval,
    x_star: np.ndarray,
    y_true: np.ndarray,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    save: bool = False,
    save_dir: str = "./",
) -> None:
    """
    Sort the predictions in ascending order of the true values \
    and plots them along with the uncertainty on the predictions and the true values.

    :param model: The sklearn like model which outputs point predictions
    :param conf_model: A MACEst model which outputs a confidence interval about some point
        prediction
    :param x_star: The set of target variables we want to predict for
    :param y_true: The true value for the target variables
    :param x_limits: x limits on figure
    :param y_limits: y limits on figure
    :param save: True means plot is saved in save_dir
    :param save_dir: Location to save figure

    :return: None
    """
    intervals = _make_prediction_intervals(conf_model, x_star)
    point_preds = model.predict(x_star)

    sorted_y, sorted_preds, sorted_intervals = _sort_predictions(
        point_preds, y_true, intervals, sort_by="y_true"
    )

    plt.figure(figsize=(12, 10))

    plt.fill_between(
        sorted_y,
        sorted_intervals.upper_95,
        sorted_intervals.lower_95,
        interpolate=True,
        alpha=0.1,
        color="b",
        label="95% confidence",
    )

    plt.fill_between(
        sorted_y,
        sorted_intervals.upper_90,
        sorted_intervals.lower_90,
        interpolate=True,
        alpha=0.2,
        color="b",
        label="90% confidence",
    )

    plt.fill_between(
        sorted_y,
        sorted_intervals.upper_70,
        sorted_intervals.lower_70,
        interpolate=True,
        alpha=0.7,
        color="b",
        label="70% confidence",
    )

    plt.scatter(sorted_y, sorted_y, marker="*", color="y", alpha=0.5, label="true")

    plt.plot(
        sorted_y,
        sorted_preds,
        color="r",
        linestyle="--",
        linewidth=3.0,
        label="prediction",
    )
    if x_limits:
        plt.xlim(x_limits)

    if y_limits:
        plt.ylim(y_limits)

    plt.ylabel("True value", fontsize=20)
    plt.xlabel("Predicted value", fontsize=20)
    plt.legend(loc="upper left")
    if save:
        plt.savefig(save_dir)


def plot_predicted_vs_true(
    model: _RegressionPointPredictionModel,
    conf_model: ModelWithPredictionInterval,
    x_star: np.ndarray,
    y_true: np.ndarray,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    save: bool = False,
    save_dir: str = "./",
) -> None:
    """
    Sorts the predictions in ascending order and plots them along with the uncertainty on the predictions \
    and the true values.

    :param model: The sklearn like model which outputs point predictions
    :param conf_model:
    :param x_star: The set of target variables we want to predict for
    :param y_true: The true value for the target variables
    :param x_limits: x limits on figure
    :param y_limits: y limits on figure
    :param save: True means plot is saved in save_dir
    :param save_dir: Location to save figure

    :return: None
    """
    intervals = _make_prediction_intervals(conf_model, x_star)

    point_preds = model.predict(x_star).flatten()

    sorted_y, sorted_preds, sorted_intervals = _sort_predictions(
        point_preds, y_true, intervals, sort_by="predictions"
    )

    plt.figure(figsize=(12, 10))

    plt.fill_between(
        sorted_preds,
        sorted_intervals.upper_95,
        sorted_intervals.lower_95,
        alpha=0.1,
        color="b",
        label="95% confidence",
    )

    plt.fill_between(
        sorted_preds,
        sorted_intervals.upper_90,
        sorted_intervals.lower_90,
        alpha=0.2,
        color="b",
        label="90% confidence",
    )

    plt.fill_between(
        sorted_preds,
        sorted_intervals.upper_70,
        sorted_intervals.lower_70,
        alpha=0.7,
        color="b",
        label="70% confidence",
    )

    plt.scatter(
        sorted_preds, sorted_y, marker="*", color="y", alpha=0.8, label="y_true"
    )

    plt.scatter(
        sorted_preds, sorted_preds, marker="o", color="r", alpha=0.7, label="prediction"
    )
    if x_limits:
        plt.xlim(x_limits)

    if y_limits:
        plt.ylim(y_limits)

    plt.xlabel("Predicted value", fontsize=20)
    plt.ylabel("Predicted value", fontsize=20)
    plt.legend(loc="upper left")
    if save:
        plt.savefig(save_dir)


def _make_prediction_intervals(
    conf_model: ModelWithPredictionInterval, x_star: np.ndarray
) -> PredictionIntervalsForPlot:
    """
    Calculate prediction intervals used for plots.

    :param conf_model: A MACEst model which outputs a confidence interval about some point
        prediction
    :param x_star: The set of target variables we want to predict

    :return:
    """
    lower_bounds = {}
    upper_bounds = {}
    for percentile in [95, 90, 70]:
        lower, upper = conf_model.predict_interval(x_star, conf_level=np.array(percentile),).T
        lower_bounds[percentile] = lower.flatten()
        upper_bounds[percentile] = upper.flatten()

    intervals = PredictionIntervalsForPlot(
        lower_95=lower_bounds[95],
        lower_90=lower_bounds[90],
        lower_70=lower_bounds[70],
        upper_95=upper_bounds[95],
        upper_90=upper_bounds[90],
        upper_70=upper_bounds[70],
    )

    return intervals


def _sort_predictions(
    point_preds: np.ndarray,
    y_true: np.ndarray,
    intervals: PredictionIntervalsForPlot,
    sort_by: Literal["predictions", "y_true"] = "predictions",
) -> SortedPredsAndIntervals:
    """
    Sort predictions in order of point predictions or the true value.

    :param point_preds: Point prediction model predictions
    :param y_true: The true value for the target variables
    :param intervals: upper and lower bounds for prediciton intervals
    :param sort_by:  plot points sorted in ascending order by either the true values, or the predicted
    :return:
    """
    if sort_by == "predictions":
        sorted_preds, sorted_y, = zip(*sorted(zip(point_preds, y_true)))

        sorted_preds, sorted_upper_95 = zip(
            *sorted(zip(point_preds, intervals.upper_95))
        )
        sorted_preds, sorted_lower_95 = zip(
            *sorted(zip(point_preds, intervals.lower_95))
        )

        sorted_preds, sorted_upper_90 = zip(
            *sorted(zip(point_preds, intervals.upper_90))
        )
        sorted_preds, sorted_lower_90 = zip(
            *sorted(zip(point_preds, intervals.lower_90))
        )

        sorted_preds, sorted_upper_70 = zip(
            *sorted(zip(point_preds, intervals.upper_70))
        )
        sorted_preds, sorted_lower_70 = zip(
            *sorted(zip(point_preds, intervals.lower_70))
        )

    elif sort_by == "y_true":
        sorted_y, sorted_preds = zip(*sorted(zip(y_true, point_preds)))

        sorted_y, sorted_upper_95 = zip(*sorted(zip(y_true, intervals.upper_95)))
        sorted_y, sorted_lower_95 = zip(*sorted(zip(y_true, intervals.lower_95)))

        sorted_y, sorted_upper_90 = zip(*sorted(zip(y_true, intervals.upper_90)))
        sorted_y, sorted_lower_90 = zip(*sorted(zip(y_true, intervals.lower_90)))

        sorted_y, sorted_upper_70 = zip(*sorted(zip(y_true, intervals.upper_70)))
        sorted_y, sorted_lower_70 = zip(*sorted(zip(y_true, intervals.lower_70)))
    else:
        raise ValueError(
            f"{sort_by} is not an option to sort by, only y_true or predictions are allowed"
        )

    sorted_intervals = PredictionIntervalsForPlot(
        lower_95=sorted_lower_95,
        lower_90=sorted_lower_90,
        lower_70=sorted_lower_70,
        upper_95=sorted_upper_95,
        upper_90=sorted_upper_90,
        upper_70=sorted_upper_70,
    )

    return SortedPredsAndIntervals(sorted_y, sorted_preds, sorted_intervals)
