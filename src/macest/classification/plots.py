# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
"""Module contains plots used for evaluating calibration methods."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import log_loss, brier_score_loss
from typing import Iterable, Union, List, Tuple

from macest.classification.utils import histogram_max_conf_pred
from macest.classification.metrics import (
    average_calibration_error,
    expected_calibration_error,
    quantile_calibration_error,
)


def plot_calibration_curve(
    confidence_scores: List[np.ndarray],
    labels: Union[List[str], Tuple[str]],
    point_predictions: Iterable[int],
    targets: Iterable[int],
    save: bool = False,
    save_dir: str = "./calibration.png",
) -> None:
    """
    Plot the (binned) emperical accuracy vs  (binned) estimated confidence for a set of predictions, The left hand \
    plot is binned by quantiles of the confidence distribution and the right hand side is binned by uniform spacing \
    in confidence. can plot n confidence scores for comparison.

    :param confidence_scores: The estimated confidence of the prediction being correct
    :param labels: The Name of the models which estimated the confidence scores
    :param point_predictions: The point prediction from the sklearn-like model
    :param targets: The true value you are trying to predict
    :param save: True means plot is saved in save_dir
    :param save_dir: Location to save figure

    :return: None
    """
    sl = np.arange(-102, 102.001, 0.1)

    nclasses = len(np.unique(targets))
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
    ax[0].set_title("Quantile calibration Curve", fontsize=20)
    ax[1].set_title("Average calibration Curve", fontsize=20)

    divider = make_axes_locatable(ax[1])
    ax_hist_x = divider.append_axes("top", 1.4, pad=0.4, sharex=ax[1])
    ax_hist_x.set_ylabel("Count")
    for i in range(2):
        ax[i].fill_between(sl, 110, sl, alpha=0.3, color="g")
        ax[i].fill_between(sl, -110, sl, alpha=0.3, color="r")

        ax[0].text(
            0.7,
            0.35,
            "Overconfident",
            fontsize=25,
            rotation=45,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[0].transAxes,
        )
        ax[0].text(
            0.35,
            0.7,
            "Underconfident",
            fontsize=25,
            rotation=45,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[0].transAxes,
        )

        ax[i].set_xlabel("Confidence (%)", fontsize=20)
        ax[i].set_ylabel("Accuracy (%)", fontsize=20)

        ax[i].plot(
            sl,
            sl,
            linestyle="--",
            linewidth=3.0,
            color=cm.Set1.colors[-5],
            label="optimal",
        )
        ax[i].set_xlim(np.array([100.0 / nclasses - 25]).clip(min=-5), 105.02)
        ax[i].set_ylim(np.array([100.0 / nclasses - 25]).clip(min=-5), 105.02)

    for i, scores in enumerate(confidence_scores):
        bins, accuracy, count, av_conf, _ = histogram_max_conf_pred(
            targets,
            point_predictions,
            point_prediction_conf=scores,
            check_conflicting_preds=False,
        )

        ax[0].plot(
            100 * av_conf,
            100 * accuracy,
            color=cm.Set1.colors[i],
            marker="o",
            markersize=12,
            label=labels[i],
            linewidth=4.0,
        )

        bins, accuracy, count, av_conf, _ = histogram_max_conf_pred(
            targets,
            point_predictions,
            point_prediction_conf=scores,
            bin_method="uniform",
        )

        ax[1].plot(
            100.0 * av_conf,
            100.0 * accuracy,
            color=cm.Set1.colors[i],
            marker="o",
            markersize=12,
            label=labels[i],
            linewidth=4.0,
        )

        ax_hist_x.hist(
            100.0 * scores,
            bins=100.0 * bins,
            color=cm.Set1.colors[i],
            label=labels[i],
            alpha=0.5,
        )

        ax_hist_x.xaxis.set_tick_params(labelbottom=False)
        ax_hist_x.set_xlabel(" ")
        ax_hist_x.legend()

    ax[0].legend()
    ax[1].legend()
    if save:
        plt.savefig(save_dir)


def plot_quantile_spaced_calibration_curve(
    confidence_scores: List[np.ndarray],
    labels: Union[List[str], Tuple[str]],
    point_predictions: Iterable[int],
    targets: Iterable[int],
    save: bool = False,
    save_dir: str = "./calibration.png",
) -> None:
    """
    Plot the (binned) emperical accuracy vs  (binned) estimated confidence for a set of predictions. The curve is \
    plotted so that points are uniformly spaced in deciles. can plot n confidence scores for comparison.

    :param confidence_scores: The estimated confidence of the prediction being correct
    :param labels: The Name of the models which estimated the confidence scores
    :param point_predictions: The point prediction from the sklearn-like model
    :param targets: The true value you are trying to predict
    :param save: True means plot is saved in save_dir
    :param save_dir: Location to save figure

    :return: None
    """
    plt.figure(figsize=(8, 8))
    plt.title("Quantile spaced calibration Curve", fontsize=20)
    nclasses = len(np.unique(targets))
    plt.xlim(np.array([100.0 / nclasses - 25]).clip(min=-5), 105.02)
    plt.ylim(np.array([100.0 / nclasses - 25]).clip(min=-5), 105.02)

    for i, scores in enumerate(confidence_scores):
        bins, accuracy, count, av_conf, _ = histogram_max_conf_pred(
            targets,
            point_predictions,
            point_prediction_conf=scores,
            check_conflicting_preds=False,
        )
        lin_ticks = np.arange(len(accuracy))
        plt.plot(
            100 * lin_ticks,
            100 * accuracy,
            color=cm.Set1.colors[i],
            marker="o",
            markersize=12,
            label=labels[i],
            linewidth=4.0,
        )

        plt.plot(
            100 * lin_ticks,
            100 * av_conf,
            linestyle="--",
            linewidth=3.0,
            color=cm.Set1.colors[i],
            label=f"optimal {labels[i]}",
        )

    plt.fill_between(
        100 * lin_ticks,
        200 * np.ones(len(accuracy)),
        100 * av_conf,
        alpha=0.3,
        color="g",
        label="underconfident",
    )
    plt.fill_between(
        100 * lin_ticks,
        -200 * np.ones(len(accuracy)),
        100 * av_conf,
        alpha=0.3,
        color="r",
        label="overconfident",
    )
    plt.xticks(100 * lin_ticks,
               labels=np.round(100 * av_conf, 2))

    plt.legend()
    if save:
        plt.savefig(save_dir)


def plot_calibration_metrics(
    confidence_scores: List[np.ndarray],
    labels: Union[List[str], Tuple[str]],
    point_predictions: Iterable[int],
    targets: Iterable[int],
    save: bool = False,
    save_dir: str = "./calibration.png",
) -> None:
    """
    Plot bar charts showing the average, expected and quantile calibration error for the confidence estimates, \
    can plot n confidence scores for comparison.

    :param confidence_scores: The estimated confidence of the prediction being correct
    :param labels: The Name of the models which estimated the confidence scores
    :param point_predictions: The point prediction from the sklearn-like model
    :param targets: The true value you are trying to predict
    :param save: True means plot is saved in save_dir
    :param save_dir: Location to save figure

    :return: None
    """
    eces = []
    aces = []
    qces = []

    for i, scores in enumerate(confidence_scores):
        eces.append(expected_calibration_error(point_predictions, targets, scores))
        aces.append(average_calibration_error(point_predictions, targets, scores))
        qces.append(quantile_calibration_error(point_predictions, targets, scores))

    fig, ax = plt.subplots(figsize=(16, 4), ncols=3, sharey=True)

    sns.barplot(x=labels, y=aces, ax=ax[0], palette=cm.Set1.colors)
    sns.barplot(x=labels, y=eces, ax=ax[1], palette=cm.Set1.colors)
    sns.barplot(x=labels, y=qces, ax=ax[2], palette=cm.Set1.colors)

    ax[0].set_ylabel("average_calibration_error")
    ax[1].set_ylabel("expected_calibration_error")
    ax[2].set_ylabel("quantile_calibration_error")
    if save:
        plt.savefig(save_dir)


def plot_forecast_metrics(
    confidence_scores: List[np.ndarray],
    labels: Union[List[str], Tuple[str]],
    point_predictions: Iterable[int],
    targets: Iterable[int],
    save: bool = False,
    save_dir: str = "./calibration.png",
) -> None:
    """
    Plot bar charts showing the negative log likelihood and the brier loss for the confidence estimates, can plot n \
    confidence scores for comparison.

    :param confidence_scores: The estimated confidence of the prediction being correct
    :param labels: The Name of the models which estimated the confidence scores
    :param point_predictions: The point prediction from the sklearn-like model
    :param targets: The true value you are trying to predict
    :param save: True means plot is saved in save_dir
    :param save_dir: Location to save figure

    :return: None
    """
    briers = []
    neg_log_loss = []
    correct = point_predictions == targets
    for i, scores in enumerate(confidence_scores):
        briers.append(brier_score_loss(correct, scores))
        neg_log_loss.append(log_loss(correct, scores))

    fig, ax = plt.subplots(figsize=(16, 4), ncols=2)

    sns.barplot(x=labels, y=neg_log_loss, ax=ax[0], palette=cm.Set1.colors)
    sns.barplot(x=labels, y=briers, ax=ax[1], palette=cm.Set1.colors)

    ax[0].set_ylabel("NLL")
    ax[1].set_ylabel("Brier loss")
    if save:
        plt.savefig(save_dir)


def plot_confidence_distribution(
    confidence_scores: np.ndarray,
    point_predictions: np.ndarray,
    targets: np.ndarray,
    split_by_correct: bool = True,
    save: bool = False,
    save_dir: str = "./calibration.png",
) -> None:
    """
    Plot the probability density function for a set of confidence scores.

    :param confidence_scores: The estimated confidence of the prediction being correct
    :param point_predictions: The point prediction from the sklearn-like model
    :param targets: The true value you are trying to predict
    :param split_by_correct: Boolean switch to determing whether to plot the full distribution of confidence scores, \
    or to plot two distributions to compare the confidence distributions when the model was correct or not.
    :param save: True means plot is saved in save_dir
    :param save_dir: Location to save figure

    :return: None
    """
    plt.figure(figsize=(10, 5))
    if split_by_correct:
        incorrect = confidence_scores[targets != point_predictions]
        correct = confidence_scores[targets == point_predictions]
        plt.title("distribution of predictions by confidence")
        sns.histplot(incorrect, kde=False, label="wrong")
        sns.histplot(correct, kde=False, label="correct")
        plt.xlabel("confidence")
        plt.legend()
    else:
        plt.title("distribution of predictions by confidence")
        sns.histplot(confidence_scores, kde=False)
        plt.xlabel("confidence")
    if save:
        plt.savefig(save_dir)
