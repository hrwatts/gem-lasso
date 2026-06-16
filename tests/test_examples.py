import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_example(name: str, **kwargs: object) -> dict[str, object]:
    namespace = runpy.run_path(str(ROOT / "examples" / name))
    return namespace["main"](**kwargs)


def test_basic_penalized_em_example_smoke() -> None:
    result = _run_example("basic_penalized_em.py")
    assert result["mode"] == "penalized_em"
    assert result["adjacency_shape"] == (2, 2)


def test_posthoc_baseline_example_smoke() -> None:
    result = _run_example("posthoc_baseline.py")
    assert result["mode"] == "posthoc"
    assert "aic" in result
    assert "bic" in result


def test_model_selection_grid_example_smoke() -> None:
    result = _run_example("model_selection_grid.py")
    assert result["n_results"] == 4
    assert result["best_mode"] == "posthoc"


def test_plot_precision_supports_generates_expected_outputs(tmp_path: Path) -> None:
    first = _run_example("plot_precision_supports.py", output_root=tmp_path / "first")
    second = _run_example("plot_precision_supports.py", output_root=tmp_path / "second")

    first_figure = tmp_path / "first" / "figures" / first["output_filenames"]["figure"]
    first_summary = tmp_path / "first" / "generated" / first["output_filenames"]["summary"]
    assert first_figure.exists()
    assert first_summary.exists()
    assert first_figure.stat().st_size > 0
    assert first_summary.stat().st_size > 0
    assert first["figure_type"] == "synthetic_precision_support"
    assert first["finite_schema_checks"]["all_panels_valid"] is True
    assert len(first["panels"]) == 2
    assert all(panel["undirected_edge_count"] >= 0 for panel in first["panels"])
    assert first == second


def test_compare_estimator_modes_generates_expected_outputs(tmp_path: Path) -> None:
    first = _run_example("compare_estimator_modes.py", output_root=tmp_path / "first")
    second = _run_example("compare_estimator_modes.py", output_root=tmp_path / "second")

    first_figure = tmp_path / "first" / "figures" / first["output_filenames"]["figure"]
    first_summary = tmp_path / "first" / "generated" / first["output_filenames"]["summary"]
    assert first_figure.exists()
    assert first_summary.exists()
    assert first_figure.stat().st_size > 0
    assert first_summary.stat().st_size > 0
    assert first["figure_type"] == "synthetic_estimator_mode_comparison"
    assert first["finite_schema_checks"]["all_scores_finite"] is True
    assert set(first["models"]) == {"penalized_em", "posthoc"}
    assert first["models"]["penalized_em"]["mode"] == "penalized_em"
    assert first["models"]["posthoc"]["mode"] == "posthoc"
    assert first["models"]["posthoc"]["likelihood_source"].startswith("dense_gaussian_mixture")
    assert all(
        model_summary["score_is_finite"] and model_summary["weights_sum_close_to_one"]
        for model_summary in first["models"].values()
    )
    assert all(
        model_summary["plot_label_alignment"]["aligned_for_plotting_only"] is True
        and model_summary["plot_label_alignment"]["reference"] == "sampled_labels"
        and set(model_summary["plot_label_alignment"]["mapping"]) == {"0", "1"}
        and 0.0 <= model_summary["plot_label_alignment"]["matched_sample_fraction"] <= 1.0
        for model_summary in first["models"].values()
    )
    assert (
        first["models"]["penalized_em"]["plot_label_alignment"]
        == second["models"]["penalized_em"]["plot_label_alignment"]
    )
    assert (
        first["models"]["posthoc"]["plot_label_alignment"]
        == second["models"]["posthoc"]["plot_label_alignment"]
    )
    assert first == second
