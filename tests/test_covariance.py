import numpy as np
import pytest

from gem_lasso.covariance import (
    GraphicalLassoResult,
    add_diagonal_jitter,
    compute_weighted_covariance,
    compute_weighted_mean,
    fit_graphical_lasso,
    is_spd,
    precision_support,
    symmetrize_matrix,
)


def test_compute_weighted_mean_matches_ml_definition() -> None:
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    weights = np.array([1.0, 3.0])
    mean = compute_weighted_mean(X, weights)
    expected = np.array([1.5, 2.5])
    assert np.allclose(mean, expected)


def test_compute_weighted_covariance_uses_ml_normalization() -> None:
    X = np.array([[0.0], [2.0]])
    weights = np.array([1.0, 3.0])
    covariance = compute_weighted_covariance(X, weights, reg_covar=0.0)
    expected = np.array([[0.75]])
    assert np.allclose(covariance, expected)


@pytest.mark.parametrize(
    "weights",
    [
        np.array([-1.0, 1.0]),
        np.array([0.0, 0.0]),
    ],
)
def test_weighted_statistics_reject_bad_weights(weights: np.ndarray) -> None:
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        compute_weighted_mean(X, weights)
    with pytest.raises(ValueError):
        compute_weighted_covariance(X, weights)


def test_weighted_covariance_applies_regularization() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    covariance = compute_weighted_covariance(X, np.array([0.5, 0.5]), reg_covar=1e-3)
    assert np.allclose(covariance, covariance.T)
    assert np.isfinite(covariance).all()
    assert np.all(np.diag(covariance) >= 1e-3)


def test_symmetrize_and_jitter_helpers() -> None:
    matrix = np.array([[1.0, 2.0], [0.0, 1.0]])
    symmetric = symmetrize_matrix(matrix)
    jittered = add_diagonal_jitter(symmetric, 0.1)
    assert np.allclose(symmetric, np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert np.allclose(jittered, np.array([[1.1, 1.0], [1.0, 1.1]]))


def test_is_spd_distinguishes_matrices() -> None:
    assert is_spd(np.array([[2.0, 0.2], [0.2, 1.0]]))
    assert not is_spd(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_precision_support_zeroes_diagonal_and_symmetrizes() -> None:
    precision = np.array(
        [
            [2.0, 0.0, 1e-4],
            [1e-5, 1.5, 0.0],
            [0.0, -2e-4, 1.2],
        ]
    )
    support = precision_support(precision, threshold=5e-5)
    expected = np.array(
        [
            [False, False, True],
            [False, False, True],
            [True, True, False],
        ]
    )
    assert np.array_equal(support, expected)


def test_precision_support_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        precision_support(np.eye(2), threshold=-1e-3)


def test_fit_graphical_lasso_returns_dataclass() -> None:
    empirical_covariance = np.array([[1.0, 0.2], [0.2, 1.5]])
    result = fit_graphical_lasso(empirical_covariance, alpha=0.1)

    assert isinstance(result, GraphicalLassoResult)
    assert np.allclose(result.covariance, result.covariance.T)
    assert np.allclose(result.precision, result.precision.T)
    assert is_spd(result.covariance)
    assert is_spd(result.precision)
    assert np.isfinite(result.covariance).all()
    assert np.isfinite(result.precision).all()
    assert result.converged is None or isinstance(result.converged, bool)


def test_fit_graphical_lasso_rejects_non_scalar_alpha() -> None:
    with pytest.raises(ValueError, match="scalar"):
        fit_graphical_lasso(np.eye(2), alpha=np.array([0.1, 0.2]))


def test_fit_graphical_lasso_tracks_jitter_path() -> None:
    singular_covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = fit_graphical_lasso(singular_covariance, alpha=0.05, jitter=1e-3)

    assert result.jitter_attempted is True
    assert result.jitter_used == pytest.approx(1e-3)
    assert is_spd(result.covariance)
    assert is_spd(result.precision)
