# Changelog

All notable changes to `gem-lasso` will be documented in this file.

The project is currently pre-release. Versioning follows the package version in
`pyproject.toml`.

## Unreleased

### Added

- clean-room Python package structure with modern `pyproject.toml` packaging
- simulation utilities for sparse precision matrices and mixture-normal sampling
- covariance helpers and graphical-lasso backend wrapper
- `GaussianMixtureGraphicalLasso` with `penalized_em` and `posthoc` modes
- model-selection helpers, runnable examples, and benchmark scaffolding
- CI workflow, release checklist, and provenance notes
- citation and public-release instrumentation (`CITATION.cff`,
  `CONTRIBUTING.md`, `SECURITY.md`)

### Validation

- editable install validation
- import smoke checks
- full local test suite
- `ruff` lint checks
- sdist and wheel build checks
