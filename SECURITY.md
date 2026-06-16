# Security Policy

## Supported versions

`gem-lasso` is currently pre-release software. Security and correctness fixes
should be assumed to target the current development snapshot rather than a
stable release line.

## Reporting a vulnerability

If you believe you have found a security issue:

1. Do not open a public issue with exploit details immediately.
2. Use GitHub private vulnerability reporting if it is enabled for the
   repository.
3. If private reporting is not available, contact the maintainer through the
   repository owner channel before disclosing details publicly.

When reporting, include:

- affected version or commit if known,
- reproduction steps,
- expected versus observed behavior,
- any proof-of-concept details needed to validate the report.

## Scope notes

For this repository, relevant security issues may include:

- unsafe dependency or build behavior,
- packaging or release artifact integrity problems,
- code paths that allow unintended execution or unsafe file handling,
- documented examples that encourage insecure usage patterns.

Numerical instability, estimator non-convergence, or benchmark regressions are
usually correctness issues rather than security issues unless they create a
clear security impact.
