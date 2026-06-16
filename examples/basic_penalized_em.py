"""Minimal penalized-EM example."""

from __future__ import annotations

import numpy as np

from gem_lasso import GaussianMixtureGraphicalLasso, sample_mixture_normal


def main() -> dict[str, object]:
    means = np.array([[0.0, 0.0], [4.0, 4.0]])
    covariances = np.stack([np.eye(2), np.array([[1.2, 0.2], [0.2, 0.8]])])
    X, _ = sample_mixture_normal(
        n_samples=120,
        weights=np.array([0.45, 0.55]),
        means=means,
        covariances=covariances,
        random_state=0,
    )

    model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.05,
        mode="penalized_em",
        max_iter=25,
        tol=1e-4,
        random_state=0,
    ).fit(X)

    adjacency = model.precision_to_adjacency(component=0)
    summary = {
        "mode": model.mode_,
        "score": model.score(X),
        "weights": model.weights_.copy(),
        "adjacency_shape": adjacency.shape,
        "n_iter": model.n_iter_,
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
