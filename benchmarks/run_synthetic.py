"""Deterministic synthetic benchmark runner for gem-lasso."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from importlib import import_module
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

render_tables_main = import_module("benchmarks.render_tables").main
_benchmark_scenarios = import_module("benchmarks.scenarios")
SyntheticScenario = _benchmark_scenarios.SyntheticScenario
get_scenarios = _benchmark_scenarios.get_scenarios
_gem_lasso = import_module("gem_lasso")
GaussianMixtureGraphicalLasso = _gem_lasso.GaussianMixtureGraphicalLasso
random_sparse_spd_matrix = _gem_lasso.random_sparse_spd_matrix
sample_mixture_normal = _gem_lasso.sample_mixture_normal
_covariance_module = import_module("gem_lasso.covariance")
compute_weighted_covariance = _covariance_module.compute_weighted_covariance
compute_weighted_mean = _covariance_module.compute_weighted_mean
fit_graphical_lasso = _covariance_module.fit_graphical_lasso
precision_support = _covariance_module.precision_support

BENCHMARK_TYPE = "small_deterministic_synthetic_snapshot"
BASELINE_ORDER = (
    "penalized_em",
    "posthoc",
    "dense_gaussian_mixture",
    "oracle_known_labels",
)
BASELINE_SEED_OFFSETS = {
    "penalized_em": 101,
    "posthoc": 211,
    "dense_gaussian_mixture": 307,
    "oracle_known_labels": 0,
}
TEST_SEED_OFFSET = 10_000
RESULTS_NAME = "synthetic_benchmark_results.csv"
SUMMARY_NAME = "synthetic_benchmark_summary.json"
MANIFEST_NAME = "synthetic_scenario_manifest.json"
SCORE_TABLE_NAME = "synthetic_benchmark_score_table.tex"
SUPPORT_TABLE_NAME = "synthetic_benchmark_support_table.tex"
CSV_FIELDNAMES = [
    "scenario_id",
    "replicate_id",
    "baseline",
    "mode",
    "alpha",
    "n_train",
    "n_test",
    "n_components",
    "n_features",
    "train_seed",
    "test_seed",
    "model_seed",
    "train_score",
    "test_score",
    "aligned_accuracy",
    "adjusted_rand_index",
    "converged",
    "n_iter",
    "warning_count",
    "fallback_count",
    "support_precision",
    "support_recall",
    "support_f1",
    "support_tp",
    "support_fp",
    "support_fn",
    "true_edge_count",
    "estimated_edge_count",
]
METRIC_DEFINITIONS = {
    "aligned_accuracy": (
        "Fraction of evaluation labels matched after deterministic permutation alignment "
        "between predicted and sampled labels."
    ),
    "adjusted_rand_index": (
        "Adjusted Rand index between predicted labels and sampled labels on the evaluation set."
    ),
    "support_precision": (
        "Precision of the undirected off-diagonal precision-support estimate after component "
        "alignment against the synthetic truth."
    ),
    "support_recall": (
        "Recall of the undirected off-diagonal precision-support estimate after component "
        "alignment against the synthetic truth."
    ),
    "support_f1": (
        "F1 score of the undirected off-diagonal precision-support estimate after component "
        "alignment against the synthetic truth."
    ),
    "true_edge_count": "Total number of true undirected off-diagonal edges across components.",
    "estimated_edge_count": (
        "Total number of estimated undirected off-diagonal edges across aligned components."
    ),
    "fallback_count": (
        "Count of component-level sparse-precision fallback events recorded during fitting."
    ),
    "warning_count": "Count of warning messages recorded during fitting.",
    "train_score": "Average observed-data log-likelihood on the synthetic training set.",
    "test_score": "Average observed-data log-likelihood on the synthetic test set.",
}


def build_benchmark_plan(scenario_set: str = "paper") -> list[SyntheticScenario]:
    """Return the named synthetic benchmark scenario set."""

    return list(get_scenarios(scenario_set))


def _align_labels(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
    *,
    n_components: int,
) -> tuple[np.ndarray, dict[int, int], int]:
    """Align candidate labels to a reference labeling for evaluation only."""

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


def _count_support_edges(support: np.ndarray) -> int:
    return int(np.count_nonzero(np.triu(support, k=1)))


def _support_counts(
    true_supports: list[np.ndarray],
    estimated_supports: list[np.ndarray],
    component_mapping: dict[int, int],
) -> dict[str, int | float]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_edges = 0
    total_estimated_edges = 0

    for estimated_component, true_component in sorted(component_mapping.items()):
        true_upper = np.triu(true_supports[true_component], k=1)
        estimated_upper = np.triu(estimated_supports[estimated_component], k=1)
        total_tp += int(np.count_nonzero(estimated_upper & true_upper))
        total_fp += int(np.count_nonzero(estimated_upper & ~true_upper))
        total_fn += int(np.count_nonzero(~estimated_upper & true_upper))
        total_true_edges += _count_support_edges(true_supports[true_component])
        total_estimated_edges += _count_support_edges(estimated_supports[estimated_component])

    support_precision = 0.0 if total_tp + total_fp == 0 else total_tp / (total_tp + total_fp)
    support_recall = 0.0 if total_tp + total_fn == 0 else total_tp / (total_tp + total_fn)
    support_f1 = (
        0.0
        if support_precision + support_recall == 0.0
        else 2.0 * support_precision * support_recall / (support_precision + support_recall)
    )
    return {
        "support_tp": total_tp,
        "support_fp": total_fp,
        "support_fn": total_fn,
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "true_edge_count": total_true_edges,
        "estimated_edge_count": total_estimated_edges,
    }


def _generate_component_parameters(
    scenario: SyntheticScenario,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int]]:
    precisions: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    edge_counts: list[int] = []
    for edge_prob, precision_seed in zip(
        scenario.edge_probs,
        scenario.precision_seeds,
        strict=True,
    ):
        precision, covariance = random_sparse_spd_matrix(
            n_features=scenario.n_features,
            edge_prob=edge_prob,
            diagonal_shift=scenario.diagonal_shift,
            random_state=precision_seed,
        )
        support = precision_support(precision)
        precisions.append(precision)
        covariances.append(covariance)
        supports.append(support)
        edge_counts.append(_count_support_edges(support))
    return np.asarray(precisions), np.asarray(covariances), supports, edge_counts


def _sample_dataset(
    scenario: SyntheticScenario,
    *,
    covariances: np.ndarray,
    seed: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    X, labels = sample_mixture_normal(
        n_samples=n_samples,
        weights=np.asarray(scenario.weights, dtype=float),
        means=np.asarray(scenario.means, dtype=float),
        covariances=covariances,
        random_state=seed,
    )
    return X, labels


def _fallback_count(model: GaussianMixtureGraphicalLasso) -> int:
    return sum(
        1
        for history_entry in model.history_
        for source in history_entry.get("glasso_fallback_source", ())
        if source is not None
    )


def _model_seed(replicate_seed: int, baseline: str) -> int:
    return int(replicate_seed + BASELINE_SEED_OFFSETS[baseline])


def _build_common_row(
    scenario: SyntheticScenario,
    *,
    replicate_seed: int,
    baseline: str,
    mode: str,
    model_seed: int | None,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "replicate_id": str(replicate_seed),
        "baseline": baseline,
        "mode": mode,
        "alpha": scenario.alpha,
        "n_train": scenario.n_train,
        "n_test": scenario.n_test,
        "n_components": scenario.n_components,
        "n_features": scenario.n_features,
        "train_seed": replicate_seed,
        "test_seed": replicate_seed + TEST_SEED_OFFSET,
        "model_seed": model_seed,
    }


def _evaluate_gem_model(
    scenario: SyntheticScenario,
    *,
    baseline: str,
    mode: str,
    model_seed: int,
    train_X: np.ndarray,
    train_labels: np.ndarray,
    test_X: np.ndarray,
    test_labels: np.ndarray,
    true_supports: list[np.ndarray],
) -> dict[str, Any]:
    model = GaussianMixtureGraphicalLasso(
        n_components=scenario.n_components,
        alpha=scenario.alpha,
        mode=mode,
        max_iter=scenario.max_iter,
        tol=scenario.tol,
        reg_covar=scenario.reg_covar,
        random_state=model_seed,
    ).fit(train_X)
    train_predicted = model.predict(train_X)
    test_predicted = model.predict(test_X)
    aligned_test, _, matched_test = _align_labels(
        test_labels,
        test_predicted,
        n_components=scenario.n_components,
    )
    _, component_mapping, _ = _align_labels(
        train_labels,
        train_predicted,
        n_components=scenario.n_components,
    )
    estimated_supports = [
        model.precision_to_adjacency(component=component, threshold=1e-8)
        for component in range(scenario.n_components)
    ]
    support_metrics = _support_counts(true_supports, estimated_supports, component_mapping)
    return {
        "train_score": float(model.score(train_X)),
        "test_score": float(model.score(test_X)),
        "aligned_accuracy": float(matched_test / test_X.shape[0]),
        "adjusted_rand_index": float(adjusted_rand_score(test_labels, aligned_test)),
        "converged": bool(model.converged_),
        "n_iter": int(model.n_iter_),
        "warning_count": int(len(model.fit_warnings_)),
        "fallback_count": int(_fallback_count(model)),
        **support_metrics,
    }


def _evaluate_dense_gmm(
    scenario: SyntheticScenario,
    *,
    model_seed: int,
    train_X: np.ndarray,
    test_X: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, Any]:
    model = GaussianMixture(
        n_components=scenario.n_components,
        covariance_type="full",
        tol=scenario.tol,
        reg_covar=scenario.reg_covar,
        max_iter=scenario.max_iter,
        init_params="kmeans",
        random_state=model_seed,
    ).fit(train_X)
    test_predicted = model.predict(test_X)
    _, _, matched_test = _align_labels(
        test_labels,
        test_predicted,
        n_components=scenario.n_components,
    )
    return {
        "train_score": float(model.score(train_X)),
        "test_score": float(model.score(test_X)),
        "aligned_accuracy": float(matched_test / test_X.shape[0]),
        "adjusted_rand_index": float(adjusted_rand_score(test_labels, test_predicted)),
        "converged": bool(model.converged_),
        "n_iter": int(model.n_iter_),
        "warning_count": 0,
        "fallback_count": 0,
        "support_precision": None,
        "support_recall": None,
        "support_f1": None,
        "support_tp": None,
        "support_fp": None,
        "support_fn": None,
        "true_edge_count": None,
        "estimated_edge_count": None,
    }


def _evaluate_oracle_known_labels(
    scenario: SyntheticScenario,
    *,
    train_X: np.ndarray,
    train_labels: np.ndarray,
    true_supports: list[np.ndarray],
) -> dict[str, Any]:
    estimated_supports: list[np.ndarray] = []
    for component in range(scenario.n_components):
        component_weights = (train_labels == component).astype(float)
        component_mean = compute_weighted_mean(train_X, component_weights)
        empirical_covariance = compute_weighted_covariance(
            train_X,
            component_weights,
            mean=component_mean,
            reg_covar=scenario.reg_covar,
        )
        result = fit_graphical_lasso(
            empirical_covariance,
            alpha=scenario.alpha,
        )
        estimated_supports.append(precision_support(result.precision))
    support_metrics = _support_counts(
        true_supports,
        estimated_supports,
        {component: component for component in range(scenario.n_components)},
    )
    return {
        "train_score": None,
        "test_score": None,
        "aligned_accuracy": None,
        "adjusted_rand_index": None,
        "converged": True,
        "n_iter": None,
        "warning_count": 0,
        "fallback_count": 0,
        **support_metrics,
    }


def _collect_rows_for_scenario(
    scenario: SyntheticScenario,
    *,
    replicate_seed: int,
    train_X: np.ndarray,
    train_labels: np.ndarray,
    test_X: np.ndarray,
    test_labels: np.ndarray,
    true_supports: list[np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    penalized_seed = _model_seed(replicate_seed, "penalized_em")
    rows.append(
        {
            **_build_common_row(
                scenario,
                replicate_seed=replicate_seed,
                baseline="penalized_em",
                mode="penalized_em",
                model_seed=penalized_seed,
            ),
            **_evaluate_gem_model(
                scenario,
                baseline="penalized_em",
                mode="penalized_em",
                model_seed=penalized_seed,
                train_X=train_X,
                train_labels=train_labels,
                test_X=test_X,
                test_labels=test_labels,
                true_supports=true_supports,
            ),
        }
    )

    posthoc_seed = _model_seed(replicate_seed, "posthoc")
    rows.append(
        {
            **_build_common_row(
                scenario,
                replicate_seed=replicate_seed,
                baseline="posthoc",
                mode="posthoc",
                model_seed=posthoc_seed,
            ),
            **_evaluate_gem_model(
                scenario,
                baseline="posthoc",
                mode="posthoc",
                model_seed=posthoc_seed,
                train_X=train_X,
                train_labels=train_labels,
                test_X=test_X,
                test_labels=test_labels,
                true_supports=true_supports,
            ),
        }
    )

    dense_seed = _model_seed(replicate_seed, "dense_gaussian_mixture")
    rows.append(
        {
            **_build_common_row(
                scenario,
                replicate_seed=replicate_seed,
                baseline="dense_gaussian_mixture",
                mode="dense_gmm",
                model_seed=dense_seed,
            ),
            **_evaluate_dense_gmm(
                scenario,
                model_seed=dense_seed,
                train_X=train_X,
                test_X=test_X,
                test_labels=test_labels,
            ),
        }
    )

    rows.append(
        {
            **_build_common_row(
                scenario,
                replicate_seed=replicate_seed,
                baseline="oracle_known_labels",
                mode="oracle_known_labels",
                model_seed=None,
            ),
            **_evaluate_oracle_known_labels(
                scenario,
                train_X=train_X,
                train_labels=train_labels,
                true_supports=true_supports,
            ),
        }
    )
    return rows


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDNAMES})


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario_id"]), str(row["baseline"]))].append(row)

    metric_names = [
        "train_score",
        "test_score",
        "aligned_accuracy",
        "adjusted_rand_index",
        "support_precision",
        "support_recall",
        "support_f1",
        "support_tp",
        "support_fp",
        "support_fn",
        "true_edge_count",
        "estimated_edge_count",
        "warning_count",
        "fallback_count",
        "n_iter",
    ]

    aggregates: list[dict[str, Any]] = []
    for scenario_id, baseline in sorted(groups):
        group_rows = groups[(scenario_id, baseline)]
        aggregate: dict[str, Any] = {
            "scenario_id": scenario_id,
            "baseline": baseline,
            "replicate_count": len(group_rows),
            "mode": group_rows[0]["mode"],
        }
        for metric_name in metric_names:
            values = [row[metric_name] for row in group_rows if row[metric_name] is not None]
            if not values:
                aggregate[f"{metric_name}_mean"] = None
                aggregate[f"{metric_name}_std"] = None
                continue
            numeric_values = np.asarray(values, dtype=float)
            aggregate[f"{metric_name}_mean"] = float(np.mean(numeric_values))
            aggregate[f"{metric_name}_std"] = float(np.std(numeric_values))
        converged_values = [
            bool(row["converged"]) for row in group_rows if row["converged"] is not None
        ]
        aggregate["all_converged"] = bool(all(converged_values)) if converged_values else None
        aggregates.append(aggregate)
    return aggregates


def _manifest(
    scenario_set: str,
    scenarios: tuple[SyntheticScenario, ...],
    scenario_edge_counts: dict[str, list[int]],
) -> dict[str, Any]:
    return {
        "script_name": "run_synthetic.py",
        "benchmark_type": BENCHMARK_TYPE,
        "scenario_set": scenario_set,
        "scenario_count": len(scenarios),
        "baseline_order": list(BASELINE_ORDER),
        "baseline_count": len(BASELINE_ORDER),
        "runtime_metrics_included": False,
        "test_seed_offset": TEST_SEED_OFFSET,
        "metric_definitions": METRIC_DEFINITIONS,
        "scenarios": [
            {
                **scenario.as_dict(),
                "true_support_edge_counts": scenario_edge_counts[scenario.scenario_id],
            }
            for scenario in scenarios
        ],
    }


def _summary(
    *,
    scenario_set: str,
    scenarios: tuple[SyntheticScenario, ...],
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    manifest_name: str,
) -> dict[str, Any]:
    score_table_rows = [
        aggregate
        for aggregate in aggregates
        if aggregate["baseline"] in {"penalized_em", "posthoc", "dense_gaussian_mixture"}
    ]
    support_table_rows = [
        aggregate
        for aggregate in aggregates
        if aggregate["baseline"] in {"penalized_em", "posthoc", "oracle_known_labels"}
    ]
    return {
        "script_name": "run_synthetic.py",
        "benchmark_type": BENCHMARK_TYPE,
        "scenario_set": scenario_set,
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "scenario_count": len(scenarios),
        "baseline_order": list(BASELINE_ORDER),
        "baseline_count": len(BASELINE_ORDER),
        "row_count": len(rows),
        "runtime_metrics_included": False,
        "metric_definitions": METRIC_DEFINITIONS,
        "manifest_filename": manifest_name,
        "output_filenames": {
            "results_csv": RESULTS_NAME,
            "summary_json": SUMMARY_NAME,
            "scenario_manifest_json": MANIFEST_NAME,
            "score_table_tex": SCORE_TABLE_NAME,
            "support_table_tex": SUPPORT_TABLE_NAME,
        },
        "aggregates": {
            "by_scenario_baseline": aggregates,
            "score_table_rows": score_table_rows,
            "support_table_rows": support_table_rows,
        },
    }


def _output_paths(output_root: str | Path | None) -> dict[str, Path]:
    root = ROOT if output_root is None else Path(output_root)
    benchmark_generated_dir = root / "benchmarks" / "generated"
    tex_generated_dir = root / "docs" / "tex" / "generated"
    benchmark_generated_dir.mkdir(parents=True, exist_ok=True)
    tex_generated_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "benchmark_generated_dir": benchmark_generated_dir,
        "tex_generated_dir": tex_generated_dir,
        "results_csv": benchmark_generated_dir / RESULTS_NAME,
        "summary_json": benchmark_generated_dir / SUMMARY_NAME,
        "scenario_manifest_json": benchmark_generated_dir / MANIFEST_NAME,
        "score_table_tex": tex_generated_dir / SCORE_TABLE_NAME,
        "support_table_tex": tex_generated_dir / SUPPORT_TABLE_NAME,
    }


def main(
    output_root: str | Path | None = None,
    *,
    scenario_set: str = "paper",
) -> dict[str, Any]:
    """Run the deterministic synthetic benchmark suite and write outputs."""

    paths = _output_paths(output_root)
    scenarios = get_scenarios(scenario_set)
    all_rows: list[dict[str, Any]] = []
    scenario_edge_counts: dict[str, list[int]] = {}

    for scenario in scenarios:
        _, covariances, true_supports, edge_counts = _generate_component_parameters(scenario)
        scenario_edge_counts[scenario.scenario_id] = edge_counts
        for replicate_seed in scenario.replicate_seeds:
            train_X, train_labels = _sample_dataset(
                scenario,
                covariances=covariances,
                seed=replicate_seed,
                n_samples=scenario.n_train,
            )
            test_X, test_labels = _sample_dataset(
                scenario,
                covariances=covariances,
                seed=replicate_seed + TEST_SEED_OFFSET,
                n_samples=scenario.n_test,
            )
            all_rows.extend(
                _collect_rows_for_scenario(
                    scenario,
                    replicate_seed=replicate_seed,
                    train_X=train_X,
                    train_labels=train_labels,
                    test_X=test_X,
                    test_labels=test_labels,
                    true_supports=true_supports,
                )
            )

    _write_csv(paths["results_csv"], all_rows)
    manifest = _manifest(scenario_set, scenarios, scenario_edge_counts)
    paths["scenario_manifest_json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    aggregates = _aggregate_rows(all_rows)
    summary = _summary(
        scenario_set=scenario_set,
        scenarios=scenarios,
        rows=all_rows,
        aggregates=aggregates,
        manifest_name=MANIFEST_NAME,
    )
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    render_tables_main(output_root=paths["root"])
    print(summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-set",
        choices=("smoke", "paper"),
        default="paper",
        help="Which deterministic scenario registry to execute.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Repository-like root directory for benchmark and TeX outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    main(output_root=arguments.output_root, scenario_set=arguments.scenario_set)
