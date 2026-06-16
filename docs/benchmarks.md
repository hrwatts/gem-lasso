# Synthetic Benchmarks

`gem-lasso` includes a small deterministic synthetic benchmark layer. It is
intended to document reproducible software behavior on generated data only. It
is not a real-data benchmark suite and should not be used to support broad
performance or superiority claims.

## Baselines

The Stage 2 synthetic benchmark runner compares:

- `GaussianMixtureGraphicalLasso(mode="penalized_em")`
- `GaussianMixtureGraphicalLasso(mode="posthoc")`
- `sklearn.mixture.GaussianMixture`
- `oracle_known_labels`, a synthetic upper-reference that fits per-component
  graphical lasso using sampled labels

The oracle row is a truth-based diagnostic only. It is not a deployable
estimator baseline.

## Scenario Sets

Two deterministic scenario sets are defined in `benchmarks/scenarios.py`.

- `smoke`: one small scenario for test-suite execution only
- `paper`: the tracked small-scope synthetic benchmark snapshot used for
  committed CSV, JSON, and TeX outputs

No public datasets are used in Stage 2.

## Metrics

The runner records the following metrics.

- `train_score`:
  average observed-data log-likelihood on the synthetic training set
- `test_score`:
  average observed-data log-likelihood on the synthetic test set
- `aligned_accuracy`:
  fraction of evaluation labels matched after deterministic permutation
  alignment between predicted labels and sampled labels
- `adjusted_rand_index`:
  adjusted Rand index between predicted labels and sampled labels on the
  synthetic test set
- `support_precision`:
  precision of the undirected off-diagonal precision-support estimate after
  component alignment against the synthetic truth
- `support_recall`:
  recall of the undirected off-diagonal precision-support estimate after
  component alignment against the synthetic truth
- `support_f1`:
  F1 score of the undirected off-diagonal precision-support estimate after
  component alignment against the synthetic truth
- `true_edge_count`:
  total number of true undirected off-diagonal edges across components
- `estimated_edge_count`:
  total number of estimated undirected off-diagonal edges across aligned
  components
- `fallback_count`:
  count of component-level sparse-precision fallback events recorded during
  fitting
- `warning_count`:
  count of warning messages recorded during fitting

Support metrics are synthetic truth-based diagnostics. They are not real-data
validation.

## Regeneration Commands

Smoke-only test run to a temporary directory:

```bash
py -3.10 benchmarks/run_synthetic.py --scenario-set smoke --output-root <tempdir>
```

Tracked paper-scenario regeneration from the repository root:

```bash
py -3.10 benchmarks/run_synthetic.py --scenario-set paper --output-root .
```

The runner writes machine-readable outputs and then renders TeX tables from the
generated summary JSON. The TeX snippets are generated artifacts and should not
be edited by hand.

## Tracked Output Files

Paper-scenario regeneration may produce the following tracked outputs:

- `benchmarks/generated/synthetic_benchmark_results.csv`
- `benchmarks/generated/synthetic_benchmark_summary.json`
- `benchmarks/generated/synthetic_scenario_manifest.json`
- `docs/tex/generated/synthetic_benchmark_score_table.tex`
- `docs/tex/generated/synthetic_benchmark_support_table.tex`

Smoke outputs should not be tracked.

## Determinism Boundary

Stage 2 committed outputs exclude wall-clock timing metrics. Runtime can be
useful for local investigation, but machine-dependent timing is intentionally
left out of the tracked deterministic benchmark snapshot.
