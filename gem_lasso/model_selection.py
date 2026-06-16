"""Model-selection helpers for gem-lasso."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mixture import GaussianMixtureGraphicalLasso


@dataclass(frozen=True)
class ModelSelectionResult:
    """Result record for one fitted estimator in a parameter grid."""

    n_components: int
    alpha: float
    mode: str
    score: float
    aic: float | None
    bic: float | None
    lower_bound: float
    converged: bool
    n_iter: int
    model: GaussianMixtureGraphicalLasso


def evaluate_model_grid(
    X: np.ndarray,
    *,
    n_components_grid: list[int] | tuple[int, ...],
    alpha_grid: list[float] | tuple[float, ...],
    mode: str = "penalized_em",
    estimator_kwargs: dict[str, Any] | None = None,
) -> list[ModelSelectionResult]:
    """Fit a small grid of models and return structured results.

    ``score`` is always recorded. ``aic`` and ``bic`` are populated only when
    the underlying estimator mode supports them, which currently means
    ``mode="posthoc"``.
    """

    component_values = _validate_n_components_grid(n_components_grid)
    alpha_values = _validate_alpha_grid(alpha_grid)
    estimator_kwargs = {} if estimator_kwargs is None else dict(estimator_kwargs)

    results: list[ModelSelectionResult] = []
    for n_components in component_values:
        for alpha in alpha_values:
            model = GaussianMixtureGraphicalLasso(
                n_components=n_components,
                alpha=alpha,
                mode=mode,
                **estimator_kwargs,
            )
            model.fit(X)
            score = float(model.score(X))
            aic_value: float | None
            bic_value: float | None
            try:
                aic_value = float(model.aic(X))
                bic_value = float(model.bic(X))
            except NotImplementedError:
                aic_value = None
                bic_value = None
            results.append(
                ModelSelectionResult(
                    n_components=n_components,
                    alpha=float(alpha),
                    mode=mode,
                    score=score,
                    aic=aic_value,
                    bic=bic_value,
                    lower_bound=float(model.lower_bound_),
                    converged=bool(model.converged_),
                    n_iter=int(model.n_iter_),
                    model=model,
                )
            )
    return results


def select_best_result(
    results: list[ModelSelectionResult] | tuple[ModelSelectionResult, ...],
    *,
    by: str = "score",
) -> ModelSelectionResult:
    """Select the best result by ``score``, ``aic``, or ``bic``."""

    if not results:
        raise ValueError("results must not be empty.")
    if by == "score":
        return max(results, key=lambda item: item.score)
    if by == "aic":
        available = [item for item in results if item.aic is not None]
        if not available:
            raise ValueError("AIC is unavailable for the provided results.")
        return min(available, key=lambda item: float(item.aic))
    if by == "bic":
        available = [item for item in results if item.bic is not None]
        if not available:
            raise ValueError("BIC is unavailable for the provided results.")
        return min(available, key=lambda item: float(item.bic))
    raise ValueError("by must be one of 'score', 'aic', or 'bic'.")


def _validate_n_components_grid(
    values: list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError("n_components_grid must not be empty.")
    if any((not isinstance(value, int)) or value < 1 for value in normalized):
        raise ValueError("n_components_grid must contain positive integers.")
    return normalized


def _validate_alpha_grid(
    values: list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError("alpha_grid must not be empty.")
    if any(value < 0.0 for value in normalized):
        raise ValueError("alpha_grid must contain nonnegative values.")
    return normalized
