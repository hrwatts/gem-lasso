"""Simulation helpers for sparse-precision Gaussian mixtures."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.utils import check_random_state

from .covariance import is_spd, symmetrize_matrix


def random_sparse_spd_matrix(
    n_features: int,
    edge_prob: float = 0.1,
    diagonal_shift: float = 0.5,
    random_state: Any = None,
    return_precision: bool = True,
    return_covariance: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate a sparse-support SPD precision matrix by construction.

    The sparse structure is imposed on the returned precision matrix. If a
    covariance matrix is requested, it is computed as the inverse of the
    precision matrix and will generally be dense.
    """

    if not isinstance(n_features, int) or n_features < 1:
        raise ValueError("n_features must be a positive integer.")
    if not 0.0 <= edge_prob <= 1.0:
        raise ValueError("edge_prob must lie in [0, 1].")
    if diagonal_shift <= 0.0:
        raise ValueError("diagonal_shift must be positive.")
    if not return_precision and not return_covariance:
        raise ValueError("At least one of return_precision or return_covariance must be True.")

    rng = check_random_state(random_state)
    support = rng.uniform(size=(n_features, n_features)) < edge_prob
    support = np.triu(support, k=1)

    magnitudes = rng.uniform(0.05, 0.35, size=(n_features, n_features))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_features, n_features))
    upper = np.triu(magnitudes * signs * support, k=1)

    precision = upper + upper.T
    diagonal = np.sum(np.abs(precision), axis=1) + diagonal_shift
    np.fill_diagonal(precision, diagonal)
    precision = symmetrize_matrix(precision)

    covariance: np.ndarray | None = None
    if return_covariance:
        covariance = symmetrize_matrix(np.linalg.inv(precision))

    if return_precision and return_covariance:
        return precision, covariance
    if return_precision:
        return precision
    return covariance


def sample_mixture_normal(
    n_samples: int,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray | None = None,
    precisions: np.ndarray | None = None,
    random_state: Any = None,
    return_labels: bool = True,
    return_posteriors: bool = False,
) -> Any:
    """Sample observations from a Gaussian mixture model.

    When ``return_posteriors=True``, the returned posterior probabilities are
    computed under the generating mixture parameters.
    """

    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")

    weights_array = np.asarray(weights, dtype=float)
    means_array = np.asarray(means, dtype=float)
    if means_array.ndim != 2:
        raise ValueError("means must be a 2D array of shape (n_components, n_features).")
    n_components, n_features = means_array.shape

    if weights_array.ndim != 1 or weights_array.shape[0] != n_components:
        raise ValueError("weights must be a 1D array matching the number of components.")
    if np.any(weights_array < 0.0):
        raise ValueError("weights must be nonnegative.")
    if not np.isclose(weights_array.sum(), 1.0, atol=1e-8):
        raise ValueError("weights must sum to 1 within tolerance.")

    provided = (covariances is not None) + (precisions is not None)
    if provided != 1:
        raise ValueError("Provide exactly one of covariances or precisions.")

    covariances_array: np.ndarray
    precisions_array: np.ndarray
    if covariances is not None:
        covariances_array = _validate_component_matrices(
            covariances,
            n_components=n_components,
            n_features=n_features,
            name="covariances",
        )
        precisions_array = np.asarray(
            [symmetrize_matrix(np.linalg.inv(component)) for component in covariances_array]
        )
    else:
        precisions_array = _validate_component_matrices(
            precisions,
            n_components=n_components,
            n_features=n_features,
            name="precisions",
        )
        covariances_array = np.asarray(
            [symmetrize_matrix(np.linalg.inv(component)) for component in precisions_array]
        )

    rng = check_random_state(random_state)
    labels = rng.choice(n_components, size=n_samples, p=weights_array)
    samples = np.empty((n_samples, n_features), dtype=float)

    for component in range(n_components):
        mask = labels == component
        count = int(mask.sum())
        if count == 0:
            continue
        samples[mask] = rng.multivariate_normal(
            mean=means_array[component],
            cov=covariances_array[component],
            size=count,
        )

    outputs: list[Any] = [samples]
    if return_labels:
        outputs.append(labels)
    if return_posteriors:
        log_joint = np.column_stack(
            [
                np.log(weights_array[component])
                + _log_gaussian_density(
                    samples,
                    means_array[component],
                    covariances_array[component],
                    precisions_array[component],
                )
                for component in range(n_components)
            ]
        )
        posteriors = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
        outputs.append(posteriors)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def _validate_component_matrices(
    matrices: np.ndarray | None,
    *,
    n_components: int,
    n_features: int,
    name: str,
) -> np.ndarray:
    matrix_array = np.asarray(matrices, dtype=float)
    if matrix_array.shape != (n_components, n_features, n_features):
        raise ValueError(
            f"{name} must have shape ({n_components}, {n_features}, {n_features})."
        )
    for component in matrix_array:
        component[:] = symmetrize_matrix(component)
        if not np.isfinite(component).all():
            raise ValueError(f"{name} must contain only finite values.")
        if not is_spd(component):
            raise ValueError(f"Each entry in {name} must be symmetric positive definite.")
    return matrix_array


def _log_gaussian_density(
    X: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    diff = X - mean
    sign, logdet_cov = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance must be positive definite.")
    quadratic = np.einsum("ni,ij,nj->n", diff, precision, diff)
    n_features = X.shape[1]
    return -0.5 * (n_features * np.log(2.0 * np.pi) + logdet_cov + quadratic)
