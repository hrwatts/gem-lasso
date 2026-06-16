"""Project-specific exceptions."""


class GemLassoError(Exception):
    """Base exception for gem-lasso."""


class GraphicalLassoFitError(GemLassoError):
    """Raised when the graphical-lasso backend fails to produce a valid fit."""
