import csv
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def _run_benchmark(output_root: Path, *, scenario_set: str = "smoke") -> dict[str, object]:
    namespace = runpy.run_path(str(ROOT / "benchmarks" / "run_synthetic.py"))
    return namespace["main"](output_root=output_root, scenario_set=scenario_set)


def _generated_paths(root: Path) -> dict[str, Path]:
    return {
        "results_csv": root / "benchmarks" / "generated" / "synthetic_benchmark_results.csv",
        "summary_json": root / "benchmarks" / "generated" / "synthetic_benchmark_summary.json",
        "manifest_json": root / "benchmarks" / "generated" / "synthetic_scenario_manifest.json",
        "score_table_tex": root
        / "docs"
        / "tex"
        / "generated"
        / "synthetic_benchmark_score_table.tex",
        "support_table_tex": root
        / "docs"
        / "tex"
        / "generated"
        / "synthetic_benchmark_support_table.tex",
    }


def test_benchmark_smoke_run_writes_expected_files(tmp_path: Path) -> None:
    summary = _run_benchmark(tmp_path / "smoke")
    paths = _generated_paths(tmp_path / "smoke")

    assert summary["scenario_set"] == "smoke"
    assert summary["scenario_count"] == 1
    assert summary["baseline_count"] == 4
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_benchmark_result_schema_is_stable(tmp_path: Path) -> None:
    _run_benchmark(tmp_path / "schema")
    paths = _generated_paths(tmp_path / "schema")

    with paths["results_csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == CSV_FIELDNAMES
    assert len(rows) == 4
    assert {row["baseline"] for row in rows} == {
        "penalized_em",
        "posthoc",
        "dense_gaussian_mixture",
        "oracle_known_labels",
    }

    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["benchmark_type"] == "small_deterministic_synthetic_snapshot"
    assert summary["runtime_metrics_included"] is False
    assert "score_table_rows" in summary["aggregates"]
    assert "support_table_rows" in summary["aggregates"]

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["scenario_set"] == "smoke"
    assert manifest["scenario_count"] == 1
    assert manifest["baseline_order"][-1] == "oracle_known_labels"
    assert "true_support_edge_counts" in manifest["scenarios"][0]


def test_benchmark_smoke_run_is_deterministic(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _run_benchmark(first_root)
    _run_benchmark(second_root)

    first_paths = _generated_paths(first_root)
    second_paths = _generated_paths(second_root)

    for key in first_paths:
        assert first_paths[key].read_text(encoding="utf-8") == second_paths[key].read_text(
            encoding="utf-8"
        )


def test_benchmark_metrics_are_finite_when_applicable(tmp_path: Path) -> None:
    _run_benchmark(tmp_path / "finite")
    paths = _generated_paths(tmp_path / "finite")

    with paths["results_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        if row["test_score"]:
            assert float(row["test_score"]) == float(row["test_score"])
        if row["aligned_accuracy"]:
            accuracy = float(row["aligned_accuracy"])
            assert 0.0 <= accuracy <= 1.0
        if row["adjusted_rand_index"]:
            ari = float(row["adjusted_rand_index"])
            assert -1.0 <= ari <= 1.0
        if row["support_precision"]:
            assert 0.0 <= float(row["support_precision"]) <= 1.0
        if row["support_recall"]:
            assert 0.0 <= float(row["support_recall"]) <= 1.0
        if row["support_f1"]:
            assert 0.0 <= float(row["support_f1"]) <= 1.0


def test_renderer_outputs_expected_labels(tmp_path: Path) -> None:
    _run_benchmark(tmp_path / "renderer")
    paths = _generated_paths(tmp_path / "renderer")
    score_table = paths["score_table_tex"].read_text(encoding="utf-8")
    support_table = paths["support_table_tex"].read_text(encoding="utf-8")

    assert r"\texttt{smoke\_k2\_d4\_sparse}" in score_table
    assert r"\texttt{penalized\_em}" in score_table
    assert r"\texttt{dense\_gaussian\_mixture}" in score_table
    assert "not real-data evidence" in score_table
    assert r"\texttt{oracle\_known\_labels}" in support_table
    assert "upper-reference" in support_table
