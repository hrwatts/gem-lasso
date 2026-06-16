"""Minimal posthoc baseline example."""

from __future__ import annotations

import numpy as np

from gem_lasso import GaussianMixtureGraphicalLasso, sample_mixture_normal


def main() -> dict[str, object]:
    means = np.array([[0.0, 0.0], [3.5, 3.5]])
    covariances = np.stack([np.eye(2), np.array([[1.5, 0.3], [0.3, 1.0]])])
    X, _ = sample_mixture_normal(
        n_samples=120,
        weights=np.array([0.5, 0.5]),
        means=means,
        covariances=covariances,
        random_state=1,
    )

    model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.05,
        mode="posthoc",
        max_iter=50,
        tol=1e-4,
        random_state=0,
    ).fit(X)

    summary = {
        "mode": model.mode_,
        "score": model.score(X),
        "aic": model.aic(X),
        "bic": model.bic(X),
        "dense_covariance_shape": model.dense_covariances_.shape,
        "sparse_covariance_shape": model.covariances_.shape,
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
