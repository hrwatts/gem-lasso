"""Structural benchmark scaffold for synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    n_samples: int
    n_features: int
    n_components: int
    alpha: float
    mode: str


def build_benchmark_plan() -> list[BenchmarkConfig]:
    """Return a small structural benchmark plan without running it."""

    return [
        BenchmarkConfig(200, 5, 2, 0.02, "penalized_em"),
        BenchmarkConfig(200, 5, 2, 0.05, "posthoc"),
        BenchmarkConfig(500, 10, 3, 0.05, "penalized_em"),
        BenchmarkConfig(500, 10, 3, 0.1, "posthoc"),
    ]


def main() -> list[BenchmarkConfig]:
    plan = build_benchmark_plan()
    for config in plan:
        print(config)
    return plan


if __name__ == "__main__":
    main()
