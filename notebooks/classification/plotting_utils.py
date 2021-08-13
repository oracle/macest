"""Module containing plotting utility functions for use in example notebooks."""
from typing import Optional, Tuple

from matplotlib.axes import Axes

from macest.classification.models import ModelWithConfidence
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

SklearnModelType = RandomForestClassifier


def plot_prediction_conf_surface(low_range: float,
                                 up_range: float,
                                 sklearn_model: SklearnModelType,
                                 X_pp_train: np.ndarray,
                                 y_pp_train: Optional[np.ndarray] = None,
                                 plot_training_data=True):
    """Plot the predictions' confidence surface."""
    x1_ne = np.random.uniform(-low_range, low_range, 10 ** 4)
    x2_ne = np.random.uniform(-low_range, low_range, 10 ** 4)
    points_ne = np.array((x1_ne, x2_ne)).T

    x1_fa = np.random.uniform(-up_range, up_range, 10 ** 4)
    x2_fa = np.random.uniform(-up_range, up_range, 10 ** 4)
    points_fa = np.array((x1_fa, x2_fa)).T

    fig, ax = plt.subplots(ncols=2, nrows=2,
                           figsize=(16, 12), )

    h0 = ax[0, 0].tricontourf(x1_ne, x2_ne, sklearn_model.predict(points_ne),
                              cmap=cm.get_cmap('rainbow', 2),
                              alpha=.9)

    g0 = ax[0, 1].tricontourf(x1_ne, x2_ne,
                              np.amax(sklearn_model.predict_proba(points_ne), axis=1),
                              vmin=0.,
                              vmax=1,
                              cmap=cm.get_cmap('viridis'))

    h1 = ax[1, 0].tricontourf(x1_fa, x2_fa, sklearn_model.predict(points_fa),
                              extend='both',
                              cmap=cm.get_cmap('rainbow', 2),
                              alpha=.9)

    g1 = ax[1, 1].tricontourf(x1_fa, x2_fa,
                              np.amax(sklearn_model.predict_proba(points_fa), axis=1),
                              vmin=0.,
                              vmax=1.,
                              cmap=cm.viridis)

    if plot_training_data:
        ax[0, 0].scatter(X_pp_train[:, 0], X_pp_train[:, 1], c=y_pp_train,
                         cmap=cm.rainbow,
                         edgecolors='black')
        ax[0, 1].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                         c=y_pp_train,
                         cmap=cm.get_cmap('rainbow', 2),
                         edgecolors='black')

    ax[1, 0].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                     c=y_pp_train,
                     edgecolors='black',
                     cmap=cm.get_cmap('rainbow', 2))

    ax[1, 1].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                     c=y_pp_train,
                     alpha=0.9,
                     cmap=cm.get_cmap('rainbow', 2),
                     edgecolors='black')

    fig.colorbar(h1, ax=[ax[0, 0], ax[1, 0]],
                 ticks=np.arange(0.0, len(np.unique(y_pp_train)) + 0.1, 1),
                 label='prediction')

    fig.colorbar(g1, ax=[ax[0, 1], ax[1, 1]],
                 ticks=np.arange(0.0, 1.01, 0.1),
                 label='Confidence')

    _set_axes_labels(ax)


def plot_prediction_conf_surface_multiclass(low_range: float,
                                            up_range: float,
                                            sklearn_model: SklearnModelType,
                                            X_pp_train: np.ndarray,
                                            y_pp_train: np.ndarray,
                                            plot_training_data=True):
    """Plot the predictions' confidence surface for the multiclass case."""
    x1_ne = np.random.uniform(-low_range, low_range, 10 ** 4)
    x2_ne = np.random.uniform(-low_range, low_range, 10 ** 4)
    points_ne = np.array((x1_ne, x2_ne)).T

    x1_fa = np.random.uniform(-up_range, up_range, 10 ** 4)
    x2_fa = np.random.uniform(-up_range, up_range, 10 ** 4)
    points_fa = np.array((x1_fa, x2_fa)).T

    fig, ax = plt.subplots(ncols=2, nrows=2,
                           figsize=(16, 12), )

    h0 = ax[0, 0].tricontourf(x1_ne, x2_ne, sklearn_model.predict(points_ne),
                              cmap=cm.get_cmap('rainbow', len(np.unique(y_pp_train))),
                              alpha=.9)

    g0 = ax[0, 1].tricontourf(x1_ne, x2_ne,
                              np.amax(sklearn_model.predict_proba(points_ne), axis=1),
                              vmin=0.,
                              vmax=1,
                              cmap=cm.get_cmap('viridis'))

    h1 = ax[1, 0].tricontourf(x1_fa, x2_fa, sklearn_model.predict(points_fa),
                              cmap=cm.get_cmap('rainbow', len(np.unique(y_pp_train))),
                              alpha=.9)

    g1 = ax[1, 1].tricontourf(x1_fa, x2_fa,
                              np.amax(sklearn_model.predict_proba(points_fa), axis=1),
                              vmin=0.,
                              vmax=1.,
                              cmap=cm.viridis)

    if plot_training_data:
        ax[0, 0].scatter(X_pp_train[:, 0], X_pp_train[:, 1], c=y_pp_train,
                         cmap=cm.rainbow,
                         edgecolors='black')
        ax[0, 1].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                         c=y_pp_train,
                         cmap=cm.get_cmap('rainbow', len(np.unique(y_pp_train))),
                         edgecolors='black')

    ax[1, 0].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                     c=y_pp_train,
                     edgecolors='black',
                     cmap=cm.get_cmap('rainbow', len(np.unique(y_pp_train))))

    ax[1, 1].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                     c=y_pp_train,
                     alpha=0.9,
                     cmap=cm.get_cmap('rainbow', 4),
                     edgecolors='black')

    fig.colorbar(h0, ax=[ax[0, 0], ax[1, 0]],
                 ticks=np.arange(0., len(np.unique(y_pp_train)) + 0.1, 1),
                 label='prediction')

    fig.colorbar(g1, ax=[ax[0, 1], ax[1, 1]],
                 ticks=np.arange(0.0, 1.01, 0.1),
                 label='Confidence')

    _set_axes_labels(ax)


def plot_macest_sklearn_comparison_surface(low_range: float,
                                           up_range: float,
                                           macest_model: ModelWithConfidence,
                                           sklearn_model: SklearnModelType,
                                           X_pp_train: Optional[np.ndarray] = None,
                                           y_pp_train: Optional[np.ndarray] = None,
                                           plot_training_data=True):
    """Plot a comparison of MACE with the original Sklearn model."""
    x1_ne = np.random.uniform(-low_range, low_range, 10 ** 4)
    x2_ne = np.random.uniform(-low_range, low_range, 10 ** 4)
    points_ne = np.array((x1_ne, x2_ne)).T

    x1_fa = np.random.uniform(-up_range, up_range, 10 ** 4)
    x2_fa = np.random.uniform(-up_range, up_range, 10 ** 4)
    points_fa = np.array((x1_fa, x2_fa)).T

    fig, ax = plt.subplots(ncols=2, nrows=2,
                           figsize=(16, 12), )

    h0 = ax[0, 0].tricontourf(x1_ne, x2_ne,
                              macest_model.predict_confidence_of_point_prediction(points_ne),
                              vmin=0.,
                              vmax=1,
                              cmap=cm.viridis)

    g0 = ax[0, 1].tricontourf(x1_ne, x2_ne, np.amax(sklearn_model.predict_proba(points_ne), axis=1),
                              vmin=0.,
                              vmax=1,
                              cmap=cm.viridis)

    if plot_training_data:
        ax[0, 0].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                         c=y_pp_train,
                         cmap=cm.rainbow,
                         edgecolors='black')

        ax[0, 1].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                         c=y_pp_train, alpha=0.9,
                         edgecolors='black',
                         cmap=cm.rainbow)

    h1 = ax[1, 0].tricontourf(x1_fa, x2_fa,
                              macest_model.predict_confidence_of_point_prediction(points_fa),
                              vmin=0.,
                              vmax=1,
                              cmap=cm.viridis)

    ax[1, 0].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                     c=y_pp_train,
                     edgecolors='black',
                     cmap=cm.get_cmap('rainbow', 2))

    g1 = ax[1, 1].tricontourf(x1_fa, x2_fa,
                              np.amax(sklearn_model.predict_proba(points_fa), axis=1),
                              vmin=0.,
                              vmax=1.,
                              cmap=cm.viridis)

    ax[1, 1].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                     c=y_pp_train,
                     alpha=0.9,
                     cmap=cm.get_cmap('rainbow', 2),
                     edgecolors='black')

    fig.colorbar(h1,
                 ax=ax.ravel().tolist(),
                 ticks=np.arange(0., 1.01, 0.1),
                 label='Confidence')

    ax[0, 0].set_title('MACEst')
    ax[1, 0].set_title('MACEst')

    ax[0, 1].set_title(f'{sklearn_model.__class__.__name__}')
    ax[1, 1].set_title(f'{sklearn_model.__class__.__name__}')

    _set_axes_labels(ax)


def plot_macest_sklearn_comparison_surface_multiclass(low_range: float,
                                                      up_range: float,
                                                      macest_model: ModelWithConfidence,
                                                      sklearn_model: SklearnModelType,
                                                      X_pp_train: np.ndarray,
                                                      y_pp_train: np.ndarray,
                                                      plot_training_data=True):
    """Plot a comparison of MACE with the original Sklearn model for the multiclass case."""
    x1_ne = np.random.uniform(-low_range, low_range, 10 ** 4)
    x2_ne = np.random.uniform(-low_range, low_range, 10 ** 4)
    points_ne = np.array((x1_ne, x2_ne)).T

    x1_fa = np.random.uniform(-up_range, up_range, 10 ** 4)
    x2_fa = np.random.uniform(-up_range, up_range, 10 ** 4)
    points_fa = np.array((x1_fa, x2_fa)).T

    fig, ax = plt.subplots(ncols=2, nrows=2,
                           figsize=(16, 12), )

    h0 = ax[0, 0].tricontourf(x1_ne, x2_ne,
                              macest_model.predict_confidence_of_point_prediction(points_ne),
                              vmin=0.,
                              vmax=1,
                              cmap=cm.viridis)

    g0 = ax[0, 1].tricontourf(x1_ne, x2_ne, np.amax(sklearn_model.predict_proba(points_ne), axis=1),
                              vmin=0.,
                              vmax=1,
                              cmap=cm.viridis)

    if plot_training_data:
        ax[0, 0].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                         c=y_pp_train,
                         cmap=cm.rainbow,
                         edgecolors='black')

        ax[0, 1].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                         c=y_pp_train, alpha=0.9,
                         edgecolors='black',
                         cmap=cm.rainbow)

    h1 = ax[1, 0].tricontourf(x1_fa, x2_fa,
                              macest_model.predict_confidence_of_point_prediction(points_fa),
                              vmin=0.,
                              vmax=1,
                              cmap=cm.viridis)

    ax[1, 0].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                     c=y_pp_train,
                     edgecolors='black',
                     cmap=cm.get_cmap('rainbow', len(np.unique(y_pp_train))))

    g1 = ax[1, 1].tricontourf(x1_fa, x2_fa,
                              np.amax(sklearn_model.predict_proba(points_fa), axis=1),
                              vmin=0.,
                              vmax=1.,
                              cmap=cm.viridis)

    ax[1, 1].scatter(X_pp_train[:, 0], X_pp_train[:, 1],
                     c=y_pp_train,
                     alpha=0.9,
                     cmap=cm.get_cmap('rainbow', len(np.unique(y_pp_train))),
                     edgecolors='black')

    fig.colorbar(h1,
                 ax=ax.ravel().tolist(),
                 ticks=np.arange(0., 1.01, 0.1),
                 label='Confidence')

    ax[0, 0].set_title('MACEst')
    ax[1, 0].set_title('MACEst')

    ax[0, 1].set_title(f'{sklearn_model.__class__.__name__}')
    ax[1, 1].set_title(f'{sklearn_model.__class__.__name__}')

    _set_axes_labels(ax)


def _set_axes_labels(ax,
                     x_label: str = "x1",
                     y_label: str = "x2",
                     fontsize: int = 18) -> None:
    ax[0, 0].set_xlabel(x_label, fontsize=fontsize)
    ax[0, 1].set_xlabel(x_label, fontsize=fontsize)

    ax[0, 0].set_ylabel(y_label, fontsize=fontsize)
    ax[0, 1].set_ylabel(y_label, fontsize=fontsize)

    ax[1, 0].set_xlabel(x_label, fontsize=fontsize)
    ax[1, 1].set_xlabel(x_label, fontsize=fontsize)

    ax[1, 0].set_ylabel(y_label, fontsize=fontsize)
    ax[1, 1].set_ylabel(y_label, fontsize=fontsize)


def make_funky_star(n_arms: int, n_points: int) -> np.ndarray:
    """Generate a star shaped data distribution."""
    arms = n_arms
    points_per_arm = int(n_points / arms)
    angs = np.arange(np.pi / arms, np.pi + 0.001, np.pi / arms)
    cov = np.array([[6, 5.93], [5.93, 6]])
    a = np.random.multivariate_normal((0, 0),
                                      cov,
                                      points_per_arm)
    arms = []
    for theta in angs:
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        arms.append(R.dot(a.T).T)

    star = np.vstack(arms)
    return star


def make_star_classes(n_arms: int, n_points: int, n_classes: int) -> np.ndarray:
    """Generate classes for the star shaped data distribution."""
    n_points_per_arm = int(n_points / n_classes)
    y = np.zeros((int(n_arms), int(n_points_per_arm)))
    for i in range(n_arms):
        j = np.mod(i, n_classes)
        y[i, :] = j * np.ones(int(n_points_per_arm))
    y = y.flatten()
    return y


def make_two_spirals(r: float,
                     n_rotations: int,
                     n_points: int,
                     noise: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate spirals for the binary classification example."""
    theta = np.linspace(0, 2 * np.pi * n_rotations, n_points)
    r0 = r * theta
    x_20 = r0 * np.cos(theta) + np.random.normal(0, noise, n_points)
    y_20 = r0 * np.sin(theta) + np.random.normal(0, noise, n_points)

    r1 = r * theta
    x_21 = - r1 * np.cos(theta) + np.random.normal(-0.4
                                                   , noise, n_points)
    y_21 = - r1 * np.sin(theta) + np.random.normal(0.4, noise, n_points)

    return x_20, x_21, y_20, y_21
