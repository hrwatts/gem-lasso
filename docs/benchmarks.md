# Benchmark Plan

This document describes the intended benchmark structure only. It does not
report results.

## Comparisons

- `GaussianMixtureGraphicalLasso(mode="penalized_em")`
- `GaussianMixtureGraphicalLasso(mode="posthoc")`
- `sklearn.mixture.GaussianMixture`
- direct graphical-lasso fits with known component assignments

## Synthetic scenarios

- low-dimensional, well-separated mixtures
- low-dimensional, overlapping mixtures
- moderate-dimensional mixtures where sparse precision recovery matters
- sensitivity sweeps over `alpha` and `n_components`

## Metrics to record when benchmarks are actually run

- fit time
- average log-likelihood on held-out data
- clustering agreement on synthetic labels
- support recovery summaries for known precision graphs
- convergence and fallback diagnostics

## Execution boundary

Do not claim benchmark outcomes in docs or release notes unless they come from
actual benchmark runs recorded separately from this scaffold.
