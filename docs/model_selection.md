# Model Selection

`gem-lasso` now includes lightweight model-selection helpers built on the
existing estimator API.

## Available helpers

- `evaluate_model_grid(...)`
- `select_best_result(...)`

These helpers are intentionally small:

- they fit a grid over `n_components` and `alpha`,
- they always record `score`,
- they record `aic` and `bic` only when the estimator mode supports them,
  which currently means `mode="posthoc"`.

## Current recommendation

- Use `mode="penalized_em"` when sparse precision updates are part of the
  fitted model itself.
- Use `mode="posthoc"` when you want dense GMM likelihood-based selection and
  sparse precision matrices only for downstream graph extraction.

## Example

```python
from gem_lasso.model_selection import evaluate_model_grid, select_best_result

results = evaluate_model_grid(
    X,
    n_components_grid=[1, 2, 3],
    alpha_grid=[0.02, 0.05],
    mode="posthoc",
    estimator_kwargs={"random_state": 0},
)

best = select_best_result(results, by="bic")
```
