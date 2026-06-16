import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_scaffold_plan_smoke() -> None:
    namespace = runpy.run_path(str(ROOT / "benchmarks" / "run_synthetic.py"))
    plan = namespace["build_benchmark_plan"]()
    assert len(plan) == 4
    assert {item.mode for item in plan} == {"penalized_em", "posthoc"}
