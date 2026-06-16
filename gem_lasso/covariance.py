"""Covariance and graphical-lasso helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import graphical_lasso as sklearn_graphical_lasso

from .exceptions import GraphicalLassoFitError


@dataclass(frozen=True)
class GraphicalLassoResult:
    """Structured return value for graphical-lasso backend calls."""

    covariance: np.ndarray
    precision: np.ndarray
    alpha: float
    n_iter: int | None
    costs: Any
    backend: str
    converged: bool | None
    jitter_used: float
    jitter_attempted: bool


def symmetrize_matrix(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    array = np.asarray(A, dtype=float)
    _validate_square_matrix(array, "A")
    return 0.5 * (array + array.T)


def add_diagonal_jitter(A: np.ndarray, jitter: float) -> np.ndarray:
    """Add diagonal jitter to a square matrix."""

    if jitter < 0.0:
        raise ValueError("jitter must be nonnegative.")
    array = symmetrize_matrix(A)
    return array + np.eye(array.shape[0], dtype=float) * jitter


def is_spd(A: np.ndarray) -> bool:
    """Return True when a matrix is symmetric positive definite."""

    array = np.asarray(A, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return False
    symmetric = np.allclose(array, array.T, atol=1e-10)
    if not symmetric or not np.isfinite(array).all():
        return False
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError:
        return False
    return True


def compute_weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute a probability-weighted mean normalized by sum(weights)."""

    observations = _validate_observations(X)
    weight_array = _validate_weights(weights, observations.shape[0])
    return np.sum(observations * weight_array[:, None], axis=0) / weight_array.sum()


def compute_weighted_covariance(
    X: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray | None = None,
    reg_covar: float = 1e-6,
) -> np.ndarray:
    """Compute an EM-style weighted covariance with ML normalization."""

    if reg_covar < 0.0:
        raise ValueError("reg_covar must be nonnegative.")
    observations = _validate_observations(X)
    weight_array = _validate_weights(weights, observations.shape[0])

    if mean is None:
        mean_array = compute_weighted_mean(observations, weight_array)
    else:
        mean_array = np.asarray(mean, dtype=float)
        if mean_array.shape != (observations.shape[1],):
            raise ValueError("mean must have shape (n_features,).")

    centered = observations - mean_array
    covariance = (centered * weight_array[:, None]).T @ centered / weight_array.sum()
    covariance = add_diagonal_jitter(covariance, reg_covar)
    return symmetrize_matrix(covariance)


def precision_support(precision: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """Return symmetric off-diagonal support from a precision matrix."""

    if threshold < 0.0:
        raise ValueError("threshold must be nonnegative.")
    precision_array = np.asarray(precision, dtype=float)
    _validate_square_matrix(precision_array, "precision")
    support = np.abs(precision_array) > threshold
    np.fill_diagonal(support, False)
    return np.logical_or(support, support.T)


def fit_graphical_lasso(
    empirical_covariance: np.ndarray,
    alpha: float,
    max_iter: int = 100,
    tol: float = 1e-4,
    jitter: float = 1e-8,
) -> GraphicalLassoResult:
    """Fit graphical lasso through a version-isolating backend wrapper."""

    if not np.isscalar(alpha):
        raise ValueError("alpha must be a scalar.")
    alpha_value = float(alpha)
    if alpha_value < 0.0:
        raise ValueError("alpha must be nonnegative.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if jitter < 0.0:
        raise ValueError("jitter must be nonnegative.")

    covariance = symmetrize_matrix(empirical_covariance)
    jitter_attempted = False
    jitter_used = 0.0

    attempt_covariance = covariance
    if not is_spd(attempt_covariance):
        if jitter == 0.0:
            raise GraphicalLassoFitError(
                "empirical_covariance is not positive definite and jitter is disabled."
            )
        attempt_covariance = add_diagonal_jitter(attempt_covariance, jitter)
        jitter_attempted = True
        jitter_used = jitter

    try:
        return _graphical_lasso_result(
            attempt_covariance,
            alpha=alpha_value,
            max_iter=max_iter,
            tol=tol,
            jitter_attempted=jitter_attempted,
            jitter_used=jitter_used,
        )
    except GraphicalLassoFitError:
        if jitter == 0.0 or jitter_attempted:
            raise
        retry_covariance = add_diagonal_jitter(covariance, jitter)
        return _graphical_lasso_result(
            retry_covariance,
            alpha=alpha_value,
            max_iter=max_iter,
            tol=tol,
            jitter_attempted=True,
            jitter_used=jitter,
        )


def _graphical_lasso_result(
    empirical_covariance: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
    jitter_attempted: bool,
    jitter_used: float,
) -> GraphicalLassoResult:
    try:
        covariance, precision, costs, n_iter = _call_graphical_lasso_backend(
            empirical_covariance,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
    except Exception as exc:  # pragma: no cover - backend detail coverage depends on sklearn
        raise GraphicalLassoFitError("Graphical lasso backend call failed.") from exc

    covariance = symmetrize_matrix(covariance)
    precision = symmetrize_matrix(precision)
    if not np.isfinite(covariance).all() or not np.isfinite(precision).all():
        raise GraphicalLassoFitError("Graphical lasso produced non-finite outputs.")
    if not is_spd(covariance) or not is_spd(precision):
        raise GraphicalLassoFitError("Graphical lasso produced a non-SPD result.")

    return GraphicalLassoResult(
        covariance=covariance,
        precision=precision,
        alpha=alpha,
        n_iter=n_iter,
        costs=costs,
        backend="sklearn.covariance.graphical_lasso",
        converged=None,
        jitter_used=jitter_used,
        jitter_attempted=jitter_attempted,
    )


def _call_graphical_lasso_backend(
    empirical_covariance: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, Any, int | None]:
    signature = inspect.signature(sklearn_graphical_lasso)
    kwargs: dict[str, Any] = {}
    if "max_iter" in signature.parameters:
        kwargs["max_iter"] = max_iter
    if "tol" in signature.parameters:
        kwargs["tol"] = tol

    expects_costs = "return_costs" in signature.parameters
    expects_n_iter = "return_n_iter" in signature.parameters
    if expects_costs:
        kwargs["return_costs"] = True
    if expects_n_iter:
        kwargs["return_n_iter"] = True

    result = sklearn_graphical_lasso(empirical_covariance, alpha, **kwargs)
    if not isinstance(result, tuple):
        raise GraphicalLassoFitError("Unexpected graphical_lasso return type.")

    if len(result) < 2:
        raise GraphicalLassoFitError("graphical_lasso returned too few values.")

    covariance = np.asarray(result[0], dtype=float)
    precision = np.asarray(result[1], dtype=float)
    costs: Any = None
    n_iter: int | None = None

    extras = list(result[2:])
    if expects_costs and extras:
        costs = extras.pop(0)
    if expects_n_iter and extras:
        n_iter = int(extras.pop(0))
    elif len(result) == 3 and not expects_costs and not expects_n_iter:
        extra = result[2]
        if np.isscalar(extra):
            n_iter = int(extra)
        else:
            costs = extra
    elif len(result) >= 4:
        costs = result[2]
        n_iter = int(result[3])

    return covariance, precision, costs, n_iter


def _validate_observations(X: np.ndarray) -> np.ndarray:
    observations = np.asarray(X, dtype=float)
    if observations.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if not np.isfinite(observations).all():
        raise ValueError("X must contain only finite values.")
    return observations


def _validate_weights(weights: np.ndarray, n_samples: int) -> np.ndarray:
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1 or weight_array.shape[0] != n_samples:
        raise ValueError("weights must be a 1D array aligned with X.")
    if np.any(weight_array < 0.0):
        raise ValueError("weights must be nonnegative.")
    if np.allclose(weight_array.sum(), 0.0):
        raise ValueError("weights must not all be zero.")
    if not np.isfinite(weight_array).all():
        raise ValueError("weights must contain only finite values.")
    return weight_array


def _validate_square_matrix(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square 2D array.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
