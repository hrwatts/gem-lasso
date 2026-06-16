"""Generate deterministic synthetic illustrations for estimator modes."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from itertools import permutations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_gem_lasso = import_module("gem_lasso")
GaussianMixtureGraphicalLasso = _gem_lasso.GaussianMixtureGraphicalLasso
sample_mixture_normal = _gem_lasso.sample_mixture_normal

DEFAULT_OUTPUT_ROOT = ROOT / "docs" / "tex"
FIGURE_NAME = "estimator_modes_shared_synthetic_dataset.png"
SUMMARY_NAME = "estimator_modes_shared_synthetic_dataset.json"


def _align_labels_for_plotting(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
    *,
    n_components: int,
) -> tuple[np.ndarray, dict[int, int], int]:
    """Align candidate labels to the reference labels for plotting only."""

    reference = np.asarray(reference_labels, dtype=int)
    candidate = np.asarray(candidate_labels, dtype=int)
    label_values = tuple(range(n_components))
    best_mapping = {label: label for label in label_values}
    best_matches = -1
    best_aligned = candidate.copy()

    for permuted_reference_labels in permutations(label_values):
        mapping = {
            predicted_label: reference_label
            for predicted_label, reference_label in zip(
                label_values,
                permuted_reference_labels,
                strict=True,
            )
        }
        aligned = np.array([mapping[int(label)] for label in candidate], dtype=int)
        matches = int(np.sum(aligned == reference))
        if matches > best_matches:
            best_mapping = mapping
            best_matches = matches
            best_aligned = aligned

    return best_aligned, best_mapping, best_matches


def main(output_root: str | Path | None = None) -> dict[str, Any]:
    """Create the estimator-mode comparison figure and JSON summary."""

    output_root_path = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    figure_dir = output_root_path / "figures"
    generated_dir = output_root_path / "generated"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    dataset_seed = 7
    penalized_seed = 3
    posthoc_seed = 5

    means = np.array(
        [
            [-2.2, -1.8, -0.5],
            [2.4, 2.0, 0.6],
        ]
    )
    covariances = np.stack(
        [
            np.array(
                [
                    [1.0, 0.35, 0.10],
                    [0.35, 0.9, 0.05],
                    [0.10, 0.05, 0.8],
                ]
            ),
            np.array(
                [
                    [0.9, -0.25, -0.15],
                    [-0.25, 1.1, 0.18],
                    [-0.15, 0.18, 0.85],
                ]
            ),
        ]
    )
    X, labels = sample_mixture_normal(
        n_samples=180,
        weights=np.array([0.48, 0.52]),
        means=means,
        covariances=covariances,
        random_state=dataset_seed,
    )

    penalized_model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.04,
        mode="penalized_em",
        max_iter=30,
        tol=1e-4,
        random_state=penalized_seed,
    ).fit(X)
    posthoc_model = GaussianMixtureGraphicalLasso(
        n_components=2,
        alpha=0.04,
        mode="posthoc",
        max_iter=30,
        tol=1e-4,
        random_state=posthoc_seed,
    ).fit(X)
    raw_penalized_labels = penalized_model.predict(X)
    raw_posthoc_labels = posthoc_model.predict(X)

    # Mixture component labels are identifiable only up to permutation, so the
    # plotted colors are aligned to the sampled labels for visual comparison.
    aligned_penalized_labels, penalized_mapping, penalized_matches = _align_labels_for_plotting(
        labels,
        raw_penalized_labels,
        n_components=penalized_model.n_components,
    )
    aligned_posthoc_labels, posthoc_mapping, posthoc_matches = _align_labels_for_plotting(
        labels,
        raw_posthoc_labels,
        n_components=posthoc_model.n_components,
    )

    cmap = ListedColormap(["#355070", "#B56576"])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    fig.suptitle(
        "Synthetic mode comparison on a shared dataset",
        fontsize=14,
    )

    panels = (
        (
            axes[0],
            labels,
            "Synthetic data\n(true sampled labels)",
            None,
        ),
        (
            axes[1],
            aligned_penalized_labels,
            "penalized_em\nsparse precision updated in-loop",
            penalized_model,
        ),
        (
            axes[2],
            aligned_posthoc_labels,
            "posthoc\ndense GMM prediction; sparse precision after fit",
            posthoc_model,
        ),
    )

    for ax, panel_labels, title, model in panels:
        scatter = ax.scatter(
            X[:, 0],
            X[:, 1],
            c=panel_labels,
            cmap=cmap,
            s=26,
            alpha=0.85,
            edgecolors="none",
        )
        del scatter
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        if model is not None:
            edge_counts = [
                int(np.count_nonzero(np.triu(model.precision_to_adjacency(component=k), k=1)))
                for k in range(model.n_components)
            ]
            ax.text(
                0.03,
                0.03,
                f"score={model.score(X):.3f}\niter={model.n_iter_}\nedges={edge_counts}",
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#666666"},
            )

    figure_path = figure_dir / FIGURE_NAME
    summary_path = generated_dir / SUMMARY_NAME
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    def _model_summary(
        model: GaussianMixtureGraphicalLasso,
        *,
        seed: int,
        likelihood_source: str,
        alignment_mapping: dict[int, int],
        aligned_matches: int,
    ) -> dict[str, Any]:
        edge_counts = [
            int(np.count_nonzero(np.triu(model.precision_to_adjacency(component=k), k=1)))
            for k in range(model.n_components)
        ]
        weights_sum = float(np.sum(model.weights_))
        score = float(model.score(X))
        return {
            "mode": model.mode_,
            "random_state": seed,
            "likelihood_source": likelihood_source,
            "score": score,
            "score_is_finite": bool(np.isfinite(score)),
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
            "n_components": int(model.n_components),
            "n_features": int(X.shape[1]),
            "adjacency_edge_counts": edge_counts,
            "weights_sum": weights_sum,
            "weights_sum_close_to_one": bool(np.isclose(weights_sum, 1.0, atol=1e-8)),
            "plot_label_alignment": {
                "reference": "sampled_labels",
                "aligned_for_plotting_only": True,
                "mapping": {
                    str(predicted_label): int(reference_label)
                    for predicted_label, reference_label in alignment_mapping.items()
                },
                "matched_sample_count": aligned_matches,
                "matched_sample_fraction": float(aligned_matches / X.shape[0]),
            },
        }

    summary = {
        "script_name": "compare_estimator_modes.py",
        "figure_type": "synthetic_estimator_mode_comparison",
        "dataset_random_state": dataset_seed,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "scatter_feature_indices": [0, 1],
        "output_filenames": {
            "figure": FIGURE_NAME,
            "summary": SUMMARY_NAME,
        },
        "models": {
            "penalized_em": _model_summary(
                penalized_model,
                seed=penalized_seed,
                likelihood_source="sparse_in_loop_estimator",
                alignment_mapping=penalized_mapping,
                aligned_matches=penalized_matches,
            ),
            "posthoc": _model_summary(
                posthoc_model,
                seed=posthoc_seed,
                likelihood_source="dense_gaussian_mixture_then_sparse_precision",
                alignment_mapping=posthoc_mapping,
                aligned_matches=posthoc_matches,
            ),
        },
        "finite_schema_checks": {
            "all_scores_finite": bool(
                np.isfinite(penalized_model.score(X)) and np.isfinite(posthoc_model.score(X))
            ),
            "all_weights_sum_close_to_one": bool(
                np.isclose(np.sum(penalized_model.weights_), 1.0, atol=1e-8)
                and np.isclose(np.sum(posthoc_model.weights_), 1.0, atol=1e-8)
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary)
    return summary


if __name__ == "__main__":
    main()
