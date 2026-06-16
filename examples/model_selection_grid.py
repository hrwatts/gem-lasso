"""Minimal model-selection example."""

from __future__ import annotations

import numpy as np

from gem_lasso import sample_mixture_normal
from gem_lasso.model_selection import evaluate_model_grid, select_best_result


def main() -> dict[str, object]:
    means = np.array([[0.0, 0.0], [4.0, 4.0]])
    covariances = np.stack([np.eye(2), np.array([[1.3, 0.2], [0.2, 0.9]])])
    X, _ = sample_mixture_normal(
        n_samples=120,
        weights=np.array([0.45, 0.55]),
        means=means,
        covariances=covariances,
        random_state=2,
    )

    results = evaluate_model_grid(
        X,
        n_components_grid=[1, 2],
        alpha_grid=[0.02, 0.05],
        mode="posthoc",
        estimator_kwargs={"random_state": 0, "max_iter": 30},
    )
    best = select_best_result(results, by="bic")
    summary = {
        "n_results": len(results),
        "best_mode": best.mode,
        "best_n_components": best.n_components,
        "best_alpha": best.alpha,
        "best_bic": best.bic,
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
