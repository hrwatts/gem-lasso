# Examples

The current example scripts are intentionally small and are safe to use as
smoke tests:

- `basic_penalized_em.py`
- `posthoc_baseline.py`
- `model_selection_grid.py`
- `plot_precision_supports.py`
- `compare_estimator_modes.py`

They demonstrate estimator setup, dense-vs-sparse posthoc separation, and the
lightweight model-selection helpers without making benchmark or recovery
claims.

## Synthetic figure regeneration

The Stage 1 figure scripts generate deterministic synthetic illustrations for
the manuscript and save both a PNG figure and a JSON summary:

Requires the plotting extra:

```bash
python -m pip install -e .[examples]
```

```bash
python examples/plot_precision_supports.py
python examples/compare_estimator_modes.py
```

Outputs are written under `docs/tex/`:

- `docs/tex/figures/`
- `docs/tex/generated/`

These assets are synthetic illustrations generated from fixed seeds. They do
not constitute benchmark evidence or public-dataset results.
