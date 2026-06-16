import numpy as np
import pytest
from scipy.special import logsumexp

from gem_lasso.simulation import random_sparse_spd_matrix, sample_mixture_normal


def _manual_posteriors(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    log_joint = []
    for component, weight in enumerate(weights):
        covariance = covariances[component]
        precision = np.linalg.inv(covariance)
        diff = X - means[component]
        sign, logdet = np.linalg.slogdet(covariance)
        assert sign > 0
        quadratic = np.einsum("ni,ij,nj->n", diff, precision, diff)
        log_density = -0.5 * (X.shape[1] * np.log(2.0 * np.pi) + logdet + quadratic)
        log_joint.append(np.log(weight) + log_density)
    stacked = np.column_stack(log_joint)
    return np.exp(stacked - logsumexp(stacked, axis=1, keepdims=True))


def test_random_sparse_spd_matrix_returns_precision_and_covariance() -> None:
    precision, covariance = random_sparse_spd_matrix(
        n_features=6,
        edge_prob=0.2,
        diagonal_shift=0.8,
        random_state=0,
    )

    assert precision.shape == (6, 6)
    assert covariance.shape == (6, 6)
    assert np.allclose(precision, precision.T)
    assert np.allclose(covariance, covariance.T)
    np.linalg.cholesky(precision)
    np.linalg.cholesky(covariance)
    assert np.allclose(covariance @ precision, np.eye(6), atol=1e-8)


def test_random_sparse_spd_matrix_covariance_is_denser_than_precision() -> None:
    precision, covariance = random_sparse_spd_matrix(
        n_features=6,
        edge_prob=0.2,
        diagonal_shift=0.8,
        random_state=7,
    )
    precision_support = np.count_nonzero(np.triu(np.abs(precision) > 1e-12, k=1))
    covariance_support = np.count_nonzero(np.triu(np.abs(covariance) > 1e-12, k=1))
    assert covariance_support >= precision_support


def test_random_sparse_spd_matrix_is_reproducible() -> None:
    first = random_sparse_spd_matrix(n_features=4, random_state=5)
    second = random_sparse_spd_matrix(n_features=4, random_state=5)

    assert isinstance(first, tuple)
    assert isinstance(second, tuple)
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_features": 0}, "n_features"),
        ({"n_features": 3, "edge_prob": -0.1}, "edge_prob"),
        ({"n_features": 3, "diagonal_shift": 0.0}, "diagonal_shift"),
        (
            {"n_features": 3, "return_precision": False, "return_covariance": False},
            "At least one",
        ),
    ],
)
def test_random_sparse_spd_matrix_rejects_bad_input(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        random_sparse_spd_matrix(**kwargs)


def test_sample_mixture_normal_returns_labels_by_default() -> None:
    covariances = np.stack([np.eye(2), np.eye(2) * 1.5])
    X, labels = sample_mixture_normal(
        n_samples=20,
        weights=np.array([0.3, 0.7]),
        means=np.array([[0.0, 0.0], [3.0, 3.0]]),
        covariances=covariances,
        random_state=0,
    )

    assert X.shape == (20, 2)
    assert labels.shape == (20,)
    assert set(np.unique(labels)).issubset({0, 1})


def test_sample_mixture_normal_accepts_precisions() -> None:
    precisions = np.stack([np.eye(2), np.eye(2) * 2.0])
    X, labels = sample_mixture_normal(
        n_samples=15,
        weights=np.array([0.5, 0.5]),
        means=np.array([[0.0, 0.0], [2.0, -2.0]]),
        precisions=precisions,
        random_state=1,
    )

    assert X.shape == (15, 2)
    assert labels.shape == (15,)


def test_sample_mixture_normal_can_return_posteriors() -> None:
    covariances = np.stack([np.eye(2), np.eye(2) * 0.5])
    weights = np.array([0.4, 0.6])
    means = np.array([[0.0, 0.0], [2.0, 2.0]])
    X, labels, posteriors = sample_mixture_normal(
        n_samples=12,
        weights=weights,
        means=means,
        covariances=covariances,
        random_state=3,
        return_posteriors=True,
    )

    expected = _manual_posteriors(X, weights, means, covariances)
    assert labels.shape == (12,)
    assert posteriors.shape == (12, 2)
    assert np.allclose(posteriors.sum(axis=1), 1.0)
    assert np.allclose(posteriors, expected)


def test_sample_mixture_normal_component_counts_match_weights_approximately() -> None:
    _, labels = sample_mixture_normal(
        n_samples=500,
        weights=np.array([0.25, 0.75]),
        means=np.array([[0.0], [2.0]]),
        covariances=np.stack([np.eye(1), np.eye(1)]),
        random_state=9,
    )
    counts = np.bincount(labels, minlength=2) / labels.size
    assert np.allclose(counts, np.array([0.25, 0.75]), atol=0.08)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "n_samples": 10,
                "weights": np.array([0.2, 0.3]),
                "means": np.array([[0.0], [1.0]]),
                "covariances": np.stack([np.eye(1), np.eye(1)]),
            },
            "sum to 1",
        ),
        (
            {
                "n_samples": 10,
                "weights": np.array([0.5, -0.5]),
                "means": np.array([[0.0], [1.0]]),
                "covariances": np.stack([np.eye(1), np.eye(1)]),
            },
            "nonnegative",
        ),
        (
            {
                "n_samples": 10,
                "weights": np.array([1.0]),
                "means": np.array([[0.0], [1.0]]),
                "covariances": np.stack([np.eye(1), np.eye(1)]),
            },
            "matching the number of components",
        ),
        (
            {
                "n_samples": 10,
                "weights": np.array([0.5, 0.5]),
                "means": np.array([[0.0], [1.0]]),
                "covariances": np.stack([np.eye(1), np.eye(1)]),
                "precisions": np.stack([np.eye(1), np.eye(1)]),
            },
            "exactly one",
        ),
    ],
)
def test_sample_mixture_normal_rejects_bad_input(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_mixture_normal(**kwargs)
