"""Generate deterministic synthetic precision-support illustrations."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

random_sparse_spd_matrix = import_module("gem_lasso").random_sparse_spd_matrix
precision_support = import_module("gem_lasso.covariance").precision_support

DEFAULT_OUTPUT_ROOT = ROOT / "docs" / "tex"
FIGURE_NAME = "precision_support_low_edge_prob_vs_high_edge_prob.png"
SUMMARY_NAME = "precision_support_low_edge_prob_vs_high_edge_prob.json"

PANEL_CONFIGS = (
    {
        "label": "Low edge probability",
        "support_label": "Sparse support",
        "edge_prob": 0.2,
        "random_state": 11,
    },
    {
        "label": "High edge probability",
        "support_label": "Denser support",
        "edge_prob": 0.8,
        "random_state": 29,
    },
)


def main(output_root: str | Path | None = None) -> dict[str, Any]:
    """Create the precision-support figure and JSON summary."""

    output_root_path = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    figure_dir = output_root_path / "figures"
    generated_dir = output_root_path / "generated"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    n_features = 12
    diagonal_shift = 0.7
    threshold = 1e-8

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    fig.suptitle(
        "Synthetic precision-support illustrations generated from fixed seeds",
        fontsize=14,
    )

    panel_summaries: list[dict[str, Any]] = []
    for ax, config in zip(axes, PANEL_CONFIGS, strict=True):
        precision, covariance = random_sparse_spd_matrix(
            n_features=n_features,
            edge_prob=config["edge_prob"],
            diagonal_shift=diagonal_shift,
            random_state=config["random_state"],
        )
        support = precision_support(precision, threshold=threshold)
        edge_count = int(np.count_nonzero(np.triu(support, k=1)))

        ax.imshow(support.astype(int), cmap="Greys", vmin=0, vmax=1)
        ax.set_title(
            f"{config['label']}\n{config['support_label']} ({edge_count} edges)",
            fontsize=11,
        )
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature index")
        ax.set_xticks(range(0, n_features, 2))
        ax.set_yticks(range(0, n_features, 2))

        panel_summaries.append(
            {
                "label": config["label"],
                "support_label": config["support_label"],
                "edge_prob": float(config["edge_prob"]),
                "random_state": int(config["random_state"]),
                "undirected_edge_count": edge_count,
                "symmetric_support": bool(np.array_equal(support, support.T)),
                "zero_diagonal": bool(not np.any(np.diag(support))),
                "finite_precision": bool(np.isfinite(precision).all()),
                "finite_covariance": bool(np.isfinite(covariance).all()),
            }
        )

    figure_path = figure_dir / FIGURE_NAME
    summary_path = generated_dir / SUMMARY_NAME
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "script_name": "plot_precision_supports.py",
        "figure_type": "synthetic_precision_support",
        "n_features": n_features,
        "diagonal_shift": diagonal_shift,
        "threshold": threshold,
        "output_filenames": {
            "figure": FIGURE_NAME,
            "summary": SUMMARY_NAME,
        },
        "finite_schema_checks": {
            "panel_count_matches_expected": len(panel_summaries) == len(PANEL_CONFIGS),
            "all_panels_valid": all(
                panel["symmetric_support"]
                and panel["zero_diagonal"]
                and panel["finite_precision"]
                and panel["finite_covariance"]
                for panel in panel_summaries
            ),
        },
        "panels": panel_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary)
    return summary


if __name__ == "__main__":
    main()
