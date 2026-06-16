# Contributing

Thanks for considering a contribution to `gem-lasso`.

## Scope

This repository is maintained as a clean-room Python implementation of Gaussian
mixture modeling with sparse precision matrices. Contributions should preserve:

- precision-based graph semantics,
- explicit numerical-safeguard behavior,
- scikit-learn-style estimator conventions where practical,
- clean-room provenance boundaries.

## Development setup

Use Python 3.10 or later.

```bash
python -m pip install -e .[dev]
```

## Before opening a change

Run the local checks:

```bash
python -m pytest
ruff check gem_lasso tests examples benchmarks
python -m build --sdist --wheel
```

If you change release-facing docs or metadata, also check:

- `README.md`
- `docs/index.md`
- `docs/release.md`
- `docs/provenance.md`

## Contribution guidelines

- Keep changes focused and scoped.
- Add or update tests for behavioral changes.
- Do not introduce copied code, comments, examples, or tests from EMgLASSO or
  other external implementations.
- Keep benchmark content structural unless actual benchmark runs are performed
  and recorded explicitly.
- Prefer documented, deterministic examples over informal notebook-only demos.

## Pull request notes

Include a short summary of:

- what changed,
- why it changed,
- how it was validated,
- whether any release-facing docs or API examples were updated.
