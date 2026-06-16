# Benchmarks

This directory contains benchmark scaffolding only. No benchmark results are
checked into the repository unless they come from actual benchmark runs.

Suggested comparisons:

- `GaussianMixtureGraphicalLasso(mode="penalized_em")`
- `GaussianMixtureGraphicalLasso(mode="posthoc")`
- `sklearn.mixture.GaussianMixture`
- direct per-component `graphical_lasso` fits on known component assignments

Suggested synthetic axes:

- sample size `n`
- feature dimension `d`
- number of components `K`
- separation between component means
- sparsity penalty `alpha`

Use the scaffold script in this directory to define configurations and record
results outside the docs before publishing any benchmark claims.
