# gem-lasso

[![CI](https://github.com/hrwatts/gem-lasso/actions/workflows/ci.yml/badge.svg)](https://github.com/hrwatts/gem-lasso/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml)
[![Package status: not on PyPI](https://img.shields.io/badge/package-not%20on%20PyPI-lightgrey)](docs/release.md)

`gem-lasso` is a clean Python implementation of Gaussian mixture modeling with
sparse component precision matrices and graph extraction.

## Current status

The current repository supports:

- simulation helpers for mixture-normal data and sparse precision generation,
- covariance and precision helper functions,
- a wrapped graphical-lasso backend interface,
- a primary penalized-EM estimator path via
  `GaussianMixtureGraphicalLasso`,
- a `posthoc` baseline mode that keeps dense-GMM likelihood and prediction in
  scikit-learn while using sparse precision estimation only for downstream
  network extraction,
- lightweight model-selection helpers,
- runnable examples and a small deterministic synthetic benchmark layer.

Deferred work now focuses on release polish, CI hardening, and any future
estimator extensions beyond the current API.

## Precision versus covariance

The sparse structure in this project lives in the precision matrix
`Omega = Sigma^{-1}`, not in the covariance matrix `Sigma`. Network edges are
read from nonzero off-diagonal precision entries. If a covariance matrix is
returned by a helper, it is generally dense even when the precision matrix is
sparse.

## Installation

Use Python 3.10 or later.

```bash
python -m pip install -e .[dev]
```

## Why gem-lasso?

If you only need dense mixture clustering, `sklearn.mixture.GaussianMixture`
already covers that use case well. `gem-lasso` is for the narrower case where
you want mixture modeling plus sparse component-level conditional-association
structure derived from precision matrices.

Relative to stitching together a dense `GaussianMixture` fit and a separate
graphical-lasso workflow by hand, `gem-lasso` gives you:

- a single estimator surface for `penalized_em` and `posthoc` workflows,
- explicit separation between dense-likelihood behavior and sparse
  network-extraction behavior,
- built-in numerical safeguards and fallback diagnostics,
- reusable simulation and model-selection helpers aligned with the estimator.

## Public API

```python
from gem_lasso import GraphicalLassoResult
from gem_lasso import GaussianMixtureGraphicalLasso
from gem_lasso import ModelSelectionResult, evaluate_model_grid, select_best_result
from gem_lasso import random_sparse_spd_matrix, sample_mixture_normal
```

## Penalized-EM estimator

The current estimator implements only the primary penalized-EM path. It fits
mixture weights, component means, and sparse precision matrices by alternating:

- an E-step using numerically stable log-density calculations and `logsumexp`,
- an M-step that reuses weighted mean/covariance helpers and fits each
  component precision matrix through the graphical-lasso backend wrapper.

The estimator exposes:

```python
from gem_lasso import GaussianMixtureGraphicalLasso

model = GaussianMixtureGraphicalLasso(
    n_components=2,
    alpha=0.05,
    mode="penalized_em",
    max_iter=50,
    tol=1e-4,
    random_state=0,
)

model.fit(X)
labels = model.predict(X)
responsibilities = model.predict_proba(X)
average_log_likelihood = model.score(X)
adjacency = model.precision_to_adjacency(component=0)
```

Expected output structure:

- `labels.shape == (n_samples,)`
- `responsibilities.shape == (n_samples, n_components)`
- `adjacency.shape == (n_features, n_features)`
- `model.mode_ == "penalized_em"`
- `model.history_` contains per-iteration likelihood and diagnostic records

## Posthoc baseline mode

The estimator also supports `mode="posthoc"` as a baseline workflow:

- `fit`, `predict`, `predict_proba`, `score`, `aic`, and `bic` are delegated to
  a dense `sklearn.mixture.GaussianMixture`,
- sparse precision matrices are estimated only after dense fitting,
- `covariances_` and `precisions_` refer to the downstream sparse
  network-extraction step,
- `dense_covariances_` preserves the dense GMM covariance estimates used for
  likelihood and prediction.

## Numerical safeguards and diagnostics

The current estimator includes practical safeguards for unstable component
updates:

- `min_effective_n` is treated as a heuristic for near-empty components, not a
  theoretical validity condition,
- near-empty components can either raise an error or reinitialize from a global
  reference estimate,
- graphical-lasso failures inside the M-step fall back to the previous
  component estimate when available, otherwise to a stabilized empirical
  covariance inverse,
- `history_` records observed log-likelihood, penalized objective, effective
  component sizes, fallback sources, and warning messages,
- `fit_warnings_` retains emitted warning text for post-fit inspection.

## Model-selection helpers

The package includes a small grid-evaluation surface:

- `evaluate_model_grid(...)`
- `select_best_result(...)`

These helpers always record `score`. They record `aic` and `bic` only when the
selected mode supports them, which currently means `mode="posthoc"`.

## Examples and benchmarks

The `examples/` directory contains runnable smoke-test examples for:

- penalized EM fitting,
- the posthoc baseline,
- grid-based model selection,
- deterministic synthetic figure generation for the manuscript.

The `benchmarks/` directory contains a small deterministic synthetic benchmark
runner, machine-readable outputs, and generated TeX table snippets. These
benchmark artifacts are synthetic only; they are not real-data evidence and do
not support broad performance claims.

Regenerate the current synthetic manuscript figures with:

Requires the plotting extra (or the full dev extra):

```bash
python -m pip install -e .[examples]
```

```bash
python examples/plot_precision_supports.py
python examples/compare_estimator_modes.py
```

Regenerate the tracked small-scope synthetic benchmark snapshot with:

```bash
python benchmarks/run_synthetic.py --scenario-set paper --output-root .
```

## Documentation

Start with:

- `docs/index.md`
- `docs/algorithm.md`
- `docs/model_selection.md`
- `docs/benchmarks.md`
- `docs/release.md`
- `docs/provenance.md`

## Validation status

The repository is validated with:

- editable-install checks,
- import smoke tests,
- unit tests for utilities, estimators, model selection, and examples,
- `ruff` linting across package, tests, examples, and benchmarks.

Before release, rerun editable-install and build validation in a fresh
environment.

## Clean-room provenance

This repository is being implemented from the mathematical description and
intended Python API behavior only. It is not a copied or translated port of the
EMgLASSO R package, and it should not reuse that package's source code,
comments, examples, or tests.
