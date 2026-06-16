import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_example(name: str) -> dict[str, object]:
    namespace = runpy.run_path(str(ROOT / "examples" / name))
    return namespace["main"]()


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
