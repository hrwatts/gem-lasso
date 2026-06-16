import numpy as np
import pytest

from gem_lasso.model_selection import (
    ModelSelectionResult,
    evaluate_model_grid,
    select_best_result,
)
from gem_lasso.simulation import sample_mixture_normal


def _dataset() -> np.ndarray:
    X, _ = sample_mixture_normal(
        n_samples=80,
        weights=np.array([0.4, 0.6]),
        means=np.array([[0.0, 0.0], [3.0, 3.0]]),
        covariances=np.stack([np.eye(2), np.array([[1.1, 0.2], [0.2, 0.9]])]),
        random_state=0,
    )
    return X


def test_evaluate_model_grid_returns_structured_results_for_penalized_em() -> None:
    results = evaluate_model_grid(
        _dataset(),
        n_components_grid=[1, 2],
        alpha_grid=[0.02, 0.05],
        mode="penalized_em",
        estimator_kwargs={"random_state": 0, "max_iter": 20},
    )

    assert len(results) == 4
    assert all(isinstance(result, ModelSelectionResult) for result in results)
    assert all(result.mode == "penalized_em" for result in results)
    assert all(result.aic is None for result in results)
    assert all(result.bic is None for result in results)


def test_evaluate_model_grid_records_aic_and_bic_for_posthoc() -> None:
    results = evaluate_model_grid(
        _dataset(),
        n_components_grid=[1, 2],
        alpha_grid=[0.02],
        mode="posthoc",
        estimator_kwargs={"random_state": 0, "max_iter": 20},
    )

    assert len(results) == 2
    assert all(result.aic is not None for result in results)
    assert all(result.bic is not None for result in results)


def test_select_best_result_supports_score_and_bic() -> None:
    results = evaluate_model_grid(
        _dataset(),
        n_components_grid=[1, 2],
        alpha_grid=[0.02],
        mode="posthoc",
        estimator_kwargs={"random_state": 0, "max_iter": 20},
    )

    best_by_score = select_best_result(results, by="score")
    best_by_bic = select_best_result(results, by="bic")
    assert best_by_score in results
    assert best_by_bic in results
    assert best_by_bic.bic == min(result.bic for result in results if result.bic is not None)


def test_select_best_result_rejects_unavailable_metric() -> None:
    results = evaluate_model_grid(
        _dataset(),
        n_components_grid=[1],
        alpha_grid=[0.02],
        mode="penalized_em",
        estimator_kwargs={"random_state": 0, "max_iter": 20},
    )

    with pytest.raises(ValueError, match="AIC"):
        select_best_result(results, by="aic")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"n_components_grid": [], "alpha_grid": [0.02]},
            "n_components_grid",
        ),
        (
            {"n_components_grid": [1], "alpha_grid": []},
            "alpha_grid",
        ),
        (
            {"n_components_grid": [0], "alpha_grid": [0.02]},
            "positive integers",
        ),
        (
            {"n_components_grid": [1], "alpha_grid": [-0.1]},
            "nonnegative",
        ),
    ],
)
def test_evaluate_model_grid_rejects_bad_grids(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_model_grid(
            _dataset(),
            mode="posthoc",
            estimator_kwargs={"random_state": 0},
            **kwargs,
        )
