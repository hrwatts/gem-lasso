import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture

from gem_lasso.covariance import GraphicalLassoResult, is_spd
from gem_lasso.exceptions import GraphicalLassoFitError
from gem_lasso.mixture import GaussianMixtureGraphicalLasso
from gem_lasso.simulation import random_sparse_spd_matrix, sample_mixture_normal


def _two_component_dataset(
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    precision_one, covariance_one = random_sparse_spd_matrix(
        n_features=2,
        edge_prob=1.0,
        diagonal_shift=1.0,
        random_state=11,
    )
    precision_two, covariance_two = random_sparse_spd_matrix(
        n_features=2,
        edge_prob=1.0,
        diagonal_shift=1.2,
        random_state=17,
    )
    means = np.array([[0.0, 0.0], [4.0, 4.0]])
    weights = np.array([0.45, 0.55])
    X, labels = sample_mixture_normal(
        n_samples=180,
        weights=weights,
        means=means,
        covariances=np.stack([covariance_one, covariance_two]),
        random_state=random_state,
    )
    return X, labels, weights, means, np.stack([precision_one, precision_two])


def test_fit_sets_expected_attributes_and_shapes() -> None:
    X, _, _, _, _ = _two_component_dataset()
    model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.05,
        max_iter=25,
        tol=1e-4,
        random_state=0,
    )

    fitted = model.fit(X)

    assert fitted is model
    assert model.weights_.shape == (2,)
    assert model.means_.shape == (2, 2)
    assert model.covariances_.shape == (2, 2, 2)
    assert model.precisions_.shape == (2, 2, 2)
    assert model.responsibilities_.shape == (X.shape[0], 2)
    assert model.labels_.shape == (X.shape[0],)
    assert model.n_iter_ >= 1
    assert isinstance(model.converged_, bool)
    assert np.isfinite(model.lower_bound_)
    assert len(model.history_) == model.n_iter_


def test_predict_proba_rows_sum_to_one_and_match_labels() -> None:
    X, _, _, _, _ = _two_component_dataset()
    model = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05, random_state=4).fit(X)

    responsibilities = model.predict_proba(X)
    labels = model.predict(X)

    assert np.allclose(responsibilities.sum(axis=1), 1.0)
    assert np.array_equal(labels, np.argmax(responsibilities, axis=1))
    assert np.array_equal(model.labels_, np.argmax(model.responsibilities_, axis=1))


def test_score_matches_final_observed_log_likelihood_per_sample() -> None:
    X, _, _, _, _ = _two_component_dataset()
    model = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05, random_state=3).fit(X)

    assert model.score(X) == pytest.approx(model.lower_bound_)
    assert model.history_[-1]["observed_log_likelihood"] / X.shape[0] == pytest.approx(
        model.lower_bound_
    )


def test_history_tracks_requested_metrics() -> None:
    X, _, _, _, _ = _two_component_dataset()
    model = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05, random_state=2).fit(X)

    entry = model.history_[0]
    assert "observed_log_likelihood" in entry
    assert "penalized_objective" in entry
    assert "delta_penalized_objective" in entry
    assert "effective_n" in entry
    assert "min_effective_n_threshold" in entry
    assert "near_empty_components" in entry
    assert "glasso_jitter_used" in entry
    assert "glasso_fallback_source" in entry
    assert "component_update_source" in entry
    assert "warning_messages" in entry
    assert np.isfinite(entry["observed_log_likelihood"])
    assert np.isfinite(entry["penalized_objective"])


def test_deterministic_under_fixed_random_state() -> None:
    X, _, _, _, _ = _two_component_dataset()
    first = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05, random_state=7).fit(X)
    second = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05, random_state=7).fit(X)

    assert np.allclose(first.weights_, second.weights_)
    assert np.allclose(first.means_, second.means_)
    assert np.allclose(first.covariances_, second.covariances_)
    assert np.allclose(first.precisions_, second.precisions_)
    assert np.allclose(first.responsibilities_, second.responsibilities_)


def test_covariances_and_precisions_are_symmetric_and_spd() -> None:
    X, _, _, _, _ = _two_component_dataset()
    model = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05, random_state=0).fit(X)

    for covariance, precision in zip(model.covariances_, model.precisions_, strict=True):
        assert np.allclose(covariance, covariance.T)
        assert np.allclose(precision, precision.T)
        assert is_spd(covariance)
        assert is_spd(precision)


def test_no_nan_likelihoods_or_probabilities() -> None:
    X, _, _, _, _ = _two_component_dataset()
    model = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05, random_state=0).fit(X)

    assert np.isfinite(model.lower_bound_)
    assert np.isfinite(model.score(X))
    assert np.isfinite(model.responsibilities_).all()
    assert np.isfinite(model.predict_proba(X)).all()


def test_precision_to_adjacency_is_symmetric_with_zero_diagonal() -> None:
    X, _, _, _, _ = _two_component_dataset()
    model = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05, random_state=0).fit(X)

    adjacency = model.precision_to_adjacency(component=0, threshold=1e-8)
    assert adjacency.shape == (2, 2)
    assert np.array_equal(adjacency, adjacency.T)
    assert not np.any(np.diag(adjacency))


def test_recovery_on_easy_two_component_example() -> None:
    X, labels, _, _, _ = _two_component_dataset(random_state=13)
    model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.02,
        max_iter=30,
        tol=1e-4,
        random_state=13,
    ).fit(X)

    predicted = model.predict(X)
    accuracy_direct = np.mean(predicted == labels)
    accuracy_swapped = np.mean((1 - predicted) == labels)
    assert max(accuracy_direct, accuracy_swapped) >= 0.85


def test_too_small_cluster_can_reinitialize() -> None:
    X, _, _, _, _ = _two_component_dataset(random_state=21)
    with pytest.warns(RuntimeWarning, match="min_effective_n heuristic"):
        model = GaussianMixtureGraphicalLasso(
            n_components=3,
            alpha=0.05,
            max_iter=10,
            min_effective_n=80,
            empty_cluster_strategy="reinitialize",
            random_state=1,
        ).fit(X)

    assert any(entry["reinitialized_components"] for entry in model.history_)
    assert any(entry["near_empty_components"] for entry in model.history_)
    assert np.allclose(model.weights_.sum(), 1.0)
    assert model.fit_warnings_


def test_too_small_cluster_can_raise_on_error_strategy() -> None:
    X, _, _, _, _ = _two_component_dataset(random_state=21)
    model = GaussianMixtureGraphicalLasso(
        n_components=3,
        alpha=0.05,
        max_iter=10,
        min_effective_n=80,
        empty_cluster_strategy="error",
        random_state=1,
    )

    with pytest.raises(ValueError, match="effective sample size"):
        model.fit(X)


def test_graphical_lasso_failure_falls_back_to_previous_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, _, _, _, _ = _two_component_dataset(random_state=5)
    mixture_module = __import__("gem_lasso.mixture", fromlist=["fit_graphical_lasso"])
    original_fit = mixture_module.fit_graphical_lasso
    call_count = {"value": 0}

    def flaky_fit(*args: object, **kwargs: object) -> GraphicalLassoResult:
        call_count["value"] += 1
        if call_count["value"] == 3:
            raise GraphicalLassoFitError("synthetic failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr("gem_lasso.mixture.fit_graphical_lasso", flaky_fit)

    with pytest.warns(RuntimeWarning, match="reusing the previous component estimate"):
        model = GaussianMixtureGraphicalLasso(
            n_components=2,
            alpha=0.05,
            max_iter=5,
            random_state=0,
        ).fit(X)

    assert any("previous component estimate" in warning for warning in model.fit_warnings_)
    assert "previous_estimate" in model.history_[0]["glasso_fallback_source"]


def test_graphical_lasso_failure_can_fall_back_to_empirical_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, _, _, _, _ = _two_component_dataset(random_state=8)

    def failing_fit(*args: object, **kwargs: object) -> GraphicalLassoResult:
        raise GraphicalLassoFitError("synthetic failure")

    monkeypatch.setattr("gem_lasso.mixture.fit_graphical_lasso", failing_fit)

    with pytest.warns(RuntimeWarning, match="empirical covariance inverse"):
        model = GaussianMixtureGraphicalLasso(
            n_components=2,
            alpha=0.05,
            max_iter=2,
            random_state=0,
        ).fit(X)

    assert any("empirical covariance inverse" in warning for warning in model.fit_warnings_)
    assert np.isfinite(model.lower_bound_)
    assert np.allclose(model.weights_.sum(), 1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_components": 0}, "n_components"),
        ({"alpha": -0.1}, "alpha"),
        ({"tol": 0.0}, "tol"),
        ({"init_params": "unsupported"}, "init_params"),
        ({"min_effective_n": -1.0}, "min_effective_n"),
        ({"empty_cluster_strategy": "drop"}, "empty_cluster_strategy"),
    ],
)
def test_bad_hyperparameters_raise(kwargs: dict, message: str) -> None:
    X, _, _, _, _ = _two_component_dataset()
    with pytest.raises(ValueError, match=message):
        GaussianMixtureGraphicalLasso(**kwargs).fit(X)


def test_bad_input_validation() -> None:
    model = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05)
    with pytest.raises(ValueError, match="2D"):
        model.fit(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="NaN|infinity"):
        model.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_predict_and_score_require_fit() -> None:
    model = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05)
    X, _, _, _, _ = _two_component_dataset()
    with pytest.raises(NotFittedError):
        model.predict(X)
    with pytest.raises(NotFittedError):
        model.predict_proba(X)
    with pytest.raises(NotFittedError):
        model.score(X)


def test_penalized_em_bic_and_aic_remain_deferred() -> None:
    X, _, _, _, _ = _two_component_dataset()
    model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.05,
        mode="penalized_em",
        random_state=0,
    ).fit(X)

    with pytest.raises(NotImplementedError, match="mode='posthoc'"):
        model.aic(X)
    with pytest.raises(NotImplementedError, match="mode='posthoc'"):
        model.bic(X)


def test_posthoc_matches_dense_gaussian_mixture_for_prediction_and_score() -> None:
    X, _, _, _, _ = _two_component_dataset(random_state=10)
    dense = GaussianMixture(
        n_components=2,
        covariance_type="full",
        tol=1e-4,
        reg_covar=1e-6,
        max_iter=100,
        init_params="kmeans",
        random_state=0,
    ).fit(X)
    model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.05,
        mode="posthoc",
        random_state=0,
    ).fit(X)

    assert np.allclose(model.predict_proba(X), dense.predict_proba(X))
    assert np.array_equal(model.predict(X), dense.predict(X))
    assert model.score(X) == pytest.approx(dense.score(X))
    assert model.aic(X) == pytest.approx(dense.aic(X))
    assert model.bic(X) == pytest.approx(dense.bic(X))


def test_posthoc_stores_dense_and_sparse_covariances_separately() -> None:
    X, _, _, _, _ = _two_component_dataset(random_state=12)
    model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.05,
        mode="posthoc",
        random_state=0,
    ).fit(X)

    assert model.dense_covariances_.shape == model.covariances_.shape
    assert not np.allclose(model.dense_covariances_, model.covariances_)
    for covariance, precision in zip(model.covariances_, model.precisions_, strict=True):
        assert np.allclose(covariance, covariance.T)
        assert np.allclose(precision, precision.T)
        assert is_spd(covariance)
        assert is_spd(precision)


def test_posthoc_history_marks_dense_baseline_separation() -> None:
    X, _, _, _, _ = _two_component_dataset(random_state=14)
    model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.05,
        mode="posthoc",
        random_state=0,
    ).fit(X)

    assert len(model.history_) == 1
    entry = model.history_[0]
    assert entry["mode"] == "posthoc"
    assert entry["dense_converged"] == model.converged_
    assert entry["dense_n_iter"] == model.n_iter_
    assert entry["dense_lower_bound"] == pytest.approx(model.lower_bound_)
    assert np.allclose(model.weights_, entry["weights"])


def test_posthoc_graph_extraction_remains_precision_based() -> None:
    X, _, _, _, _ = _two_component_dataset(random_state=15)
    model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.05,
        mode="posthoc",
        random_state=0,
    ).fit(X)

    adjacency = model.precision_to_adjacency(component=1, threshold=1e-8)
    assert adjacency.shape == (2, 2)
    assert np.array_equal(adjacency, adjacency.T)
    assert not np.any(np.diag(adjacency))


def test_posthoc_graphical_lasso_failure_uses_empirical_inverse_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, _, _, _, _ = _two_component_dataset(random_state=16)

    def failing_fit(*args: object, **kwargs: object) -> GraphicalLassoResult:
        raise GraphicalLassoFitError("synthetic failure")

    monkeypatch.setattr("gem_lasso.mixture.fit_graphical_lasso", failing_fit)

    with pytest.warns(RuntimeWarning, match="empirical covariance inverse"):
        model = GaussianMixtureGraphicalLasso(
            n_components=2,
            alpha=0.05,
            mode="posthoc",
            max_iter=5,
            random_state=0,
        ).fit(X)

    assert any("empirical covariance inverse" in warning for warning in model.fit_warnings_)
    assert np.isfinite(model.score(X))


@pytest.mark.parametrize("mode", ["penalized_em", "posthoc"])
def test_mode_hyperparameter_accepts_supported_values(mode: str) -> None:
    X, _, _, _, _ = _two_component_dataset()
    model = GaussianMixtureGraphicalLasso(n_components=2, alpha=0.05, mode=mode, random_state=0)
    fitted = model.fit(X)
    assert fitted.mode_ == mode


def test_bad_mode_hyperparameter_raises() -> None:
    X, _, _, _, _ = _two_component_dataset()
    with pytest.raises(ValueError, match="mode"):
        GaussianMixtureGraphicalLasso(
            n_components=2,
            alpha=0.05,
            mode="unsupported",
            random_state=0,
        ).fit(X)
