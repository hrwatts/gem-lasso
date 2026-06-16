# Algorithm Notes

The current implementation includes simulation utilities, covariance helpers,
the graphical-lasso backend wrapper, a primary penalized-EM estimator path, and
a posthoc dense-GMM baseline mode.

## Covariance and precision

For a Gaussian random vector with covariance matrix `Sigma`, the precision
matrix is `Omega = Sigma^{-1}`. Sparse conditional association structure is
represented in the off-diagonal entries of `Omega`, not in the entries of
`Sigma`.

## Sparse precision support

The simulation utilities generate sparse precision matrices directly. Their
inverse covariance matrices will usually be dense.

## Graphical-lasso backend

The graphical-lasso wrapper in `gem_lasso.covariance` centralizes the
interaction with scikit-learn so solver-specific behavior stays isolated from
the rest of the package.

## Penalized-EM path

`GaussianMixtureGraphicalLasso` reuses the Phase 0-1 helpers rather than
duplicating them:

- weighted means come from `compute_weighted_mean`,
- weighted empirical covariances come from `compute_weighted_covariance`,
- sparse precision updates go through `fit_graphical_lasso`,
- graph adjacency is extracted from precision support only.

The estimator records both observed-data log-likelihood and an internal
penalized objective in `history_`. Its convergence flag is based on a
documented tolerance check on changes in that penalized objective per sample;
it should not be interpreted as a proof of monotone convergence.

## Numerical safeguards

The current estimator treats numerical recovery as an explicit part of the
algorithm surface:

- `min_effective_n` is a heuristic threshold for near-empty components,
- when `empty_cluster_strategy="reinitialize"`, those components are reseeded
  from a reference estimate built on the full dataset,
- if a component-level graphical-lasso fit fails, the estimator first falls
  back to the previous component estimate when one exists,
- when no previous component estimate exists, it falls back to a stabilized
  empirical covariance inverse constructed with diagonal jitter escalation.

These events are recorded in `history_`, and warning text is also collected in
`fit_warnings_` for downstream diagnostics.

## Posthoc baseline mode

`mode="posthoc"` uses `sklearn.mixture.GaussianMixture` as the authoritative
dense model for:

- responsibilities,
- predicted labels,
- observed-data log-likelihood via `score`,
- model selection criteria via `aic` and `bic`.

After dense fitting, sparse precision matrices are estimated separately from
the final responsibilities. This preserves a strict separation between dense
mixture likelihood behavior and downstream sparse network extraction.
