# Benchmarks

This directory contains the deterministic synthetic benchmark runner and
scenario registry for Stage 2. The benchmark layer is synthetic only. It does
not use public datasets and it should not be used to support real-data or broad
superiority claims.

## Commands

Smoke scenario set, for tests or local validation:

```bash
python benchmarks/run_synthetic.py --scenario-set smoke --output-root <tempdir>
```

Paper scenario set, for tracked deterministic outputs:

```bash
python benchmarks/run_synthetic.py --scenario-set paper --output-root .
```

## Output Files

The runner writes:

- `benchmarks/generated/synthetic_benchmark_results.csv`
- `benchmarks/generated/synthetic_benchmark_summary.json`
- `benchmarks/generated/synthetic_scenario_manifest.json`
- `docs/tex/generated/synthetic_benchmark_score_table.tex`
- `docs/tex/generated/synthetic_benchmark_support_table.tex`

Smoke outputs should remain in temporary directories only.
