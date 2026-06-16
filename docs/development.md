# Development

Use Python 3.10 or later.

## Setup

```bash
uv sync --python 3.10 --extra dev
```

## Validation

```bash
uv run --python 3.10 python -m pip install -e .[dev]
uv run --python 3.10 pytest
uv run --python 3.10 ruff check .
```

Use developer-facing notes for local environment issues. Do not place
machine-specific setup details in the public README.
