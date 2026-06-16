"""Deterministic synthetic benchmark scenarios for gem-lasso."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticScenario:
    """A deterministic synthetic benchmark configuration."""

    scenario_id: str
    description: str
    n_train: int
    n_test: int
    n_components: int
    n_features: int
    weights: tuple[float, ...]
    means: tuple[tuple[float, ...], ...]
    edge_probs: tuple[float, ...]
    precision_seeds: tuple[int, ...]
    diagonal_shift: float
    alpha: float
    replicate_seeds: tuple[int, ...]
    max_iter: int = 40
    tol: float = 1e-4
    reg_covar: float = 1e-6

    def __post_init__(self) -> None:
        if len(self.weights) != self.n_components:
            raise ValueError("weights must match n_components.")
        if len(self.means) != self.n_components:
            raise ValueError("means must match n_components.")
        if any(len(mean) != self.n_features for mean in self.means):
            raise ValueError("each mean vector must match n_features.")
        if len(self.edge_probs) != self.n_components:
            raise ValueError("edge_probs must match n_components.")
        if len(self.precision_seeds) != self.n_components:
            raise ValueError("precision_seeds must match n_components.")
        if len(self.replicate_seeds) < 1:
            raise ValueError("replicate_seeds must be non-empty.")

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_components": self.n_components,
            "n_features": self.n_features,
            "weights": list(self.weights),
            "means": [list(mean) for mean in self.means],
            "edge_probs": list(self.edge_probs),
            "precision_seeds": list(self.precision_seeds),
            "diagonal_shift": self.diagonal_shift,
            "alpha": self.alpha,
            "replicate_seeds": list(self.replicate_seeds),
            "max_iter": self.max_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
        }


SMOKE_SCENARIOS: tuple[SyntheticScenario, ...] = (
    SyntheticScenario(
        scenario_id="smoke_k2_d4_sparse",
        description="Small deterministic smoke scenario for benchmark regression tests.",
        n_train=120,
        n_test=320,
        n_components=2,
        n_features=4,
        weights=(0.48, 0.52),
        means=(
            (-1.9, -1.3, 0.2, 0.6),
            (1.8, 1.4, -0.3, -0.5),
        ),
        edge_probs=(0.15, 0.25),
        precision_seeds=(101, 202),
        diagonal_shift=0.6,
        alpha=0.05,
        replicate_seeds=(7001,),
    ),
)


PAPER_SCENARIOS: tuple[SyntheticScenario, ...] = (
    SyntheticScenario(
        scenario_id="separated_k2_d5_sparse",
        description="Two-component low-dimensional mixture with clear separation.",
        n_train=240,
        n_test=900,
        n_components=2,
        n_features=5,
        weights=(0.5, 0.5),
        means=(
            (-2.5, -1.9, 0.3, 0.8, 1.1),
            (2.4, 1.8, -0.4, -0.7, -1.2),
        ),
        edge_probs=(0.15, 0.2),
        precision_seeds=(11, 29),
        diagonal_shift=0.7,
        alpha=0.05,
        replicate_seeds=(8101, 8102, 8103),
    ),
    SyntheticScenario(
        scenario_id="overlapping_k2_d5_sparse",
        description="Two-component low-dimensional mixture with stronger overlap.",
        n_train=240,
        n_test=900,
        n_components=2,
        n_features=5,
        weights=(0.5, 0.5),
        means=(
            (-1.2, -0.8, 0.0, 0.3, 0.4),
            (1.0, 0.9, 0.1, -0.2, -0.3),
        ),
        edge_probs=(0.15, 0.2),
        precision_seeds=(41, 53),
        diagonal_shift=0.7,
        alpha=0.05,
        replicate_seeds=(8201, 8202, 8203),
    ),
    SyntheticScenario(
        scenario_id="moderate_k3_d8_sparse",
        description="Three-component moderate-dimensional sparse-support scenario.",
        n_train=360,
        n_test=1200,
        n_components=3,
        n_features=8,
        weights=(0.34, 0.31, 0.35),
        means=(
            (-2.2, -1.7, 0.2, 0.8, 0.4, -0.3, 0.6, -0.1),
            (0.4, 1.3, 1.6, -0.5, 0.0, 0.8, -0.2, 0.5),
            (2.0, -0.2, 2.1, 0.9, -0.6, 0.2, 1.2, -0.7),
        ),
        edge_probs=(0.1, 0.15, 0.2),
        precision_seeds=(71, 83, 97),
        diagonal_shift=0.8,
        alpha=0.05,
        replicate_seeds=(8301, 8302),
    ),
)


def get_scenarios(scenario_set: str) -> tuple[SyntheticScenario, ...]:
    """Return the named benchmark scenario set."""

    if scenario_set == "smoke":
        return SMOKE_SCENARIOS
    if scenario_set == "paper":
        return PAPER_SCENARIOS
    raise ValueError("scenario_set must be one of {'smoke', 'paper'}.")
