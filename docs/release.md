# Release Checklist

This checklist is for release-facing validation only.

## Fresh-environment validation

- create or select a clean Python 3.10+ environment
- install with `python -m pip install -e .[dev]`
- run import smoke checks against the public API
- run the full test suite
- run `ruff` across package, tests, examples, and benchmarks

## Packaging checks

- build both sdist and wheel
- inspect build artifacts for expected package contents
- confirm README renders as intended from packaged metadata

## API review

- confirm the README public API examples match the exported symbols
- confirm `posthoc` documentation preserves dense-vs-sparse separation
- confirm no deferred or unsupported behavior is described as implemented

## Provenance review

- confirm clean-room wording remains accurate
- confirm no copied R code, comments, examples, or tests were introduced
- confirm benchmark docs remain structural unless real benchmark runs are recorded

## CI expectations

- editable install
- import smoke
- full pytest run
- `ruff` lint
- build check
