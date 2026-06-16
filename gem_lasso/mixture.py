"""Penalized-EM estimator for Gaussian mixtures with sparse precision matrices."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from .covariance import (
    GraphicalLassoResult,
    add_diagonal_jitter,
    compute_weighted_covariance,
    compute_weighted_mean,
    fit_graphical_lasso,
    is_spd,
    precision_support,
    symmetrize_matrix,
)
from .exceptions import GraphicalLassoFitError


@dataclass(frozen=True)
class _MixtureParameters:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    precisions: np.ndarray


class GaussianMixtureGraphicalLasso(BaseEstimator, ClusterMixin):
    """Gaussian mixture estimator with graphical-lasso precision updates.

    The estimator supports two workflows:

    - ``mode="penalized_em"`` updates sparse precision matrices inside the EM
      loop.
    - ``mode="posthoc"`` fits a dense :class:`sklearn.mixture.GaussianMixture`
      for likelihood and prediction, then applies graphical lasso only as a
      downstream network-extraction step.

    In ``penalized_em`` mode, the estimator tracks both observed-data
    log-likelihood and an internal penalized objective in ``history_``. The
    convergence flag is based on the absolute change in that penalized
    objective per sample and should not be interpreted as a claim of monotone
    convergence.
    """

    def __init__(
        self,
        n_components: int = 1,
        alpha: float = 0.05,
        mode: str = "penalized_em",
        tol: float = 1e-4,
        max_iter: int = 100,
        init_params: str = "kmeans",
        reg_covar: float = 1e-6,
        min_effective_n: float | None = None,
        empty_cluster_strategy: str = "reinitialize",
        glasso_max_iter: int = 100,
        glasso_tol: float = 1e-4,
        covariance_jitter: float = 1e-8,
        random_state: Any = None,
        verbose: int = 0,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.mode = mode
        self.tol = tol
        self.max_iter = max_iter
        self.init_params = init_params
        self.reg_covar = reg_covar
        self.min_effective_n = min_effective_n
        self.empty_cluster_strategy = empty_cluster_strategy
        self.glasso_max_iter = glasso_max_iter
        self.glasso_tol = glasso_tol
        self.covariance_jitter = covariance_jitter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: None = None) -> "GaussianMixtureGraphicalLasso":
        """Fit the estimator in the configured mode."""

        del y
        X_array = self._validate_X(X)
        self._validate_hyperparameters(X_array)
        self.n_features_in_ = X_array.shape[1]
        rng = check_random_state(self.random_state)
        self.mode_ = self.mode

        if self.mode == "posthoc":
            self._fit_posthoc(X_array, rng)
            return self

        self._fit_penalized_em(X_array, rng)
        return self

    def aic(self, X: np.ndarray) -> float:
        """Return AIC for ``mode="posthoc"`` only.

        In posthoc mode the dense GaussianMixture controls likelihood and model
        selection criteria; sparse precision estimates are downstream network
        extraction artifacts. Approximate penalized-EM information criteria
        remain deferred.
        """

        check_is_fitted(self, ["mode_"])
        X_array = self._validate_X(X)
        if self.mode_ != "posthoc":
            raise NotImplementedError(
                "aic is implemented only for mode='posthoc'; approximate "
                "penalized_em information criteria remain deferred."
            )
        return float(self._dense_model_.aic(X_array))

    def bic(self, X: np.ndarray) -> float:
        """Return BIC for ``mode="posthoc"`` only.

        In posthoc mode the dense GaussianMixture controls likelihood and model
        selection criteria; sparse precision estimates are downstream network
        extraction artifacts. Approximate penalized-EM information criteria
        remain deferred.
        """

        check_is_fitted(self, ["mode_"])
        X_array = self._validate_X(X)
        if self.mode_ != "posthoc":
            raise NotImplementedError(
                "bic is implemented only for mode='posthoc'; approximate "
                "penalized_em information criteria remain deferred."
            )
        return float(self._dense_model_.bic(X_array))

    def _fit_penalized_em(
        self,
        X_array: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        parameters, initialization_meta = self._initialize_parameters(X_array, rng)
        history: list[dict[str, Any]] = []
        prev_penalized_objective: float | None = None
        final_observed_log_likelihood: float | None = None
        final_responsibilities: np.ndarray | None = None
        converged = False
        fit_warnings = list(initialization_meta["warning_messages"])

        for iteration in range(1, self.max_iter + 1):
            responsibilities, _ = self._estimate_responsibilities(X_array, parameters)
            parameters, update_meta = self._m_step(
                X_array,
                responsibilities,
                rng,
                previous_parameters=parameters,
                iteration=iteration,
            )
            fit_warnings.extend(update_meta["warning_messages"])
            evaluated_responsibilities, observed_log_likelihood = self._estimate_responsibilities(
                X_array, parameters
            )
            penalized_objective = self._compute_penalized_objective(
                observed_log_likelihood=observed_log_likelihood,
                responsibilities=evaluated_responsibilities,
                precisions=parameters.precisions,
            )
            delta_penalized = (
                None
                if prev_penalized_objective is None
                else penalized_objective - prev_penalized_objective
            )
            history.append(
                {
                    "iteration": iteration,
                    "observed_log_likelihood": observed_log_likelihood,
                    "penalized_objective": penalized_objective,
                    "delta_penalized_objective": delta_penalized,
                    "effective_n": update_meta["effective_n"].copy(),
                    "weights": parameters.weights.copy(),
                    "min_effective_n_threshold": update_meta["min_effective_n_threshold"],
                    "near_empty_components": tuple(update_meta["near_empty_components"]),
                    "reinitialized_components": tuple(update_meta["reinitialized_components"]),
                    "glasso_n_iter": tuple(update_meta["glasso_n_iter"]),
                    "glasso_jitter_used": tuple(update_meta["glasso_jitter_used"]),
                    "glasso_fallback_source": tuple(update_meta["glasso_fallback_source"]),
                    "component_update_source": tuple(update_meta["component_update_source"]),
                    "warning_messages": tuple(update_meta["warning_messages"]),
                }
            )

            final_responsibilities = evaluated_responsibilities
            final_observed_log_likelihood = observed_log_likelihood
            if (
                prev_penalized_objective is not None
                and abs(delta_penalized) / X_array.shape[0] <= self.tol
            ):
                converged = True
                break
            prev_penalized_objective = penalized_objective

        self.weights_ = parameters.weights
        self.means_ = parameters.means
        self.covariances_ = parameters.covariances
        self.precisions_ = parameters.precisions
        self.responsibilities_ = final_responsibilities
        self.labels_ = np.argmax(final_responsibilities, axis=1)
        self.converged_ = converged
        self.n_iter_ = len(history)
        self.lower_bound_ = float(final_observed_log_likelihood / X_array.shape[0])
        self.history_ = history
        self.fit_warnings_ = fit_warnings
        if hasattr(self, "_dense_model_"):
            del self._dense_model_
        if hasattr(self, "dense_covariances_"):
            del self.dense_covariances_

    def _fit_posthoc(
        self,
        X_array: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        dense_model = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            init_params=self.init_params,
            random_state=rng,
            verbose=self.verbose,
        )
        dense_model.fit(X_array)
        responsibilities = dense_model.predict_proba(X_array)
        labels = dense_model.predict(X_array)
        parameters, update_meta = self._update_sparse_parameters(
            X_array,
            responsibilities,
            rng,
            previous_parameters=None,
            iteration=0,
            means_override=dense_model.means_,
        )
        observed_log_likelihood = float(dense_model.score(X_array) * X_array.shape[0])
        penalized_objective = self._compute_penalized_objective(
            observed_log_likelihood=observed_log_likelihood,
            responsibilities=responsibilities,
            precisions=parameters.precisions,
        )

        self.weights_ = dense_model.weights_.copy()
        self.means_ = dense_model.means_.copy()
        self.covariances_ = parameters.covariances
        self.precisions_ = parameters.precisions
        self.dense_covariances_ = dense_model.covariances_.copy()
        self.responsibilities_ = responsibilities
        self.labels_ = labels
        self.converged_ = bool(dense_model.converged_)
        self.n_iter_ = int(dense_model.n_iter_)
        self.lower_bound_ = float(dense_model.lower_bound_)
        self.history_ = [
            {
                "iteration": 0,
                "mode": "posthoc",
                "observed_log_likelihood": observed_log_likelihood,
                "penalized_objective": penalized_objective,
                "delta_penalized_objective": None,
                "effective_n": update_meta["effective_n"].copy(),
                "weights": self.weights_.copy(),
                "min_effective_n_threshold": update_meta["min_effective_n_threshold"],
                "near_empty_components": tuple(update_meta["near_empty_components"]),
                "reinitialized_components": tuple(update_meta["reinitialized_components"]),
                "glasso_n_iter": tuple(update_meta["glasso_n_iter"]),
                "glasso_jitter_used": tuple(update_meta["glasso_jitter_used"]),
                "glasso_fallback_source": tuple(update_meta["glasso_fallback_source"]),
                "component_update_source": tuple(update_meta["component_update_source"]),
                "warning_messages": tuple(update_meta["warning_messages"]),
                "dense_converged": bool(dense_model.converged_),
                "dense_n_iter": int(dense_model.n_iter_),
                "dense_lower_bound": float(dense_model.lower_bound_),
            }
        ]
        self.fit_warnings_ = list(update_meta["warning_messages"])
        self._dense_model_ = dense_model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior responsibilities for each component."""

        check_is_fitted(self, ["weights_", "means_", "covariances_", "precisions_", "mode_"])
        X_array = self._validate_X(X)
        if self.mode_ == "posthoc":
            return self._dense_model_.predict_proba(X_array)
        responsibilities, _ = self._estimate_responsibilities(X_array, self._fitted_parameters())
        return responsibilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each observation to its highest-responsibility component."""

        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray) -> float:
        """Return the average observed-data log-likelihood."""

        check_is_fitted(self, ["weights_", "means_", "covariances_", "precisions_", "mode_"])
        X_array = self._validate_X(X)
        if self.mode_ == "posthoc":
            return float(self._dense_model_.score(X_array))
        _, observed_log_likelihood = self._estimate_responsibilities(
            X_array, self._fitted_parameters()
        )
        return float(observed_log_likelihood / X_array.shape[0])

    def precision_to_adjacency(self, component: int, threshold: float = 1e-8) -> np.ndarray:
        """Return a symmetric adjacency mask from a fitted precision matrix."""

        check_is_fitted(self, ["precisions_"])
        if component < 0 or component >= self.precisions_.shape[0]:
            raise IndexError("component is out of range.")
        return precision_support(self.precisions_[component], threshold=threshold)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        check_array_kwargs: dict[str, Any] = {"dtype": float, "ensure_2d": True}
        if "ensure_all_finite" in inspect.signature(check_array).parameters:
            check_array_kwargs["ensure_all_finite"] = True
        else:
            check_array_kwargs["force_all_finite"] = True
        return check_array(X, **check_array_kwargs)

    def _validate_hyperparameters(self, X: np.ndarray) -> None:
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if X.shape[0] < self.n_components:
            raise ValueError("n_samples must be at least n_components.")
        if not np.isscalar(self.alpha) or float(self.alpha) < 0.0:
            raise ValueError("alpha must be a nonnegative scalar.")
        if self.mode not in {"penalized_em", "posthoc"}:
            raise ValueError("mode must be either 'penalized_em' or 'posthoc'.")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive.")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if self.init_params not in {"kmeans", "random"}:
            raise ValueError("init_params must be either 'kmeans' or 'random'.")
        if self.reg_covar < 0.0:
            raise ValueError("reg_covar must be nonnegative.")
        if self.min_effective_n is not None and self.min_effective_n < 0.0:
            raise ValueError("min_effective_n must be nonnegative when provided.")
        if self.empty_cluster_strategy not in {"reinitialize", "error"}:
            raise ValueError("empty_cluster_strategy must be 'reinitialize' or 'error'.")
        if not isinstance(self.glasso_max_iter, int) or self.glasso_max_iter < 1:
            raise ValueError("glasso_max_iter must be a positive integer.")
        if self.glasso_tol <= 0.0:
            raise ValueError("glasso_tol must be positive.")
        if self.covariance_jitter < 0.0:
            raise ValueError("covariance_jitter must be nonnegative.")
        if not isinstance(self.verbose, int) or self.verbose < 0:
            raise ValueError("verbose must be a nonnegative integer.")

    def _initialize_parameters(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> tuple[_MixtureParameters, dict[str, Any]]:
        if self.init_params == "kmeans":
            kmeans = KMeans(n_clusters=self.n_components, n_init=10, random_state=rng)
            labels = kmeans.fit_predict(X)
            responsibilities = np.zeros((X.shape[0], self.n_components), dtype=float)
            responsibilities[np.arange(X.shape[0]), labels] = 1.0
        else:
            responsibilities = rng.dirichlet(np.ones(self.n_components), size=X.shape[0])
        return self._m_step(X, responsibilities, rng, previous_parameters=None, iteration=0)

    def _m_step(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray,
        rng: np.random.RandomState,
        *,
        previous_parameters: _MixtureParameters | None,
        iteration: int,
    ) -> tuple[_MixtureParameters, dict[str, Any]]:
        return self._update_sparse_parameters(
            X,
            responsibilities,
            rng,
            previous_parameters=previous_parameters,
            iteration=iteration,
            means_override=None,
        )

    def _update_sparse_parameters(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray,
        rng: np.random.RandomState,
        *,
        previous_parameters: _MixtureParameters | None,
        iteration: int,
        means_override: np.ndarray | None,
    ) -> tuple[_MixtureParameters, dict[str, Any]]:
        effective_n = responsibilities.sum(axis=0)
        means = np.empty((self.n_components, X.shape[1]), dtype=float)
        covariances = np.empty((self.n_components, X.shape[1], X.shape[1]), dtype=float)
        precisions = np.empty((self.n_components, X.shape[1], X.shape[1]), dtype=float)
        weights = np.empty(self.n_components, dtype=float)

        near_empty_components: list[int] = []
        reinitialized_components: list[int] = []
        glasso_n_iter: list[int | None] = []
        glasso_jitter_used: list[float] = []
        glasso_fallback_source: list[str | None] = []
        component_update_source: list[str] = []
        warning_messages: list[str] = []
        global_result: GraphicalLassoResult | None = None
        min_effective_n = self._effective_n_threshold(X.shape[1])
        weight_floor = np.finfo(float).tiny
        means_override_array: np.ndarray | None = None

        if means_override is not None:
            means_override_array = np.asarray(means_override, dtype=float)
            if means_override_array.shape != (self.n_components, X.shape[1]):
                raise ValueError(
                    "means_override must have shape "
                    f"({self.n_components}, {X.shape[1]})."
                )

        for component in range(self.n_components):
            component_weights = responsibilities[:, component]
            if effective_n[component] <= min_effective_n:
                near_empty_components.append(component)
                if self.empty_cluster_strategy == "error":
                    raise ValueError(
                        "Component "
                        f"{component} effective sample size {effective_n[component]:.6g} "
                        f"fell below the min_effective_n heuristic {min_effective_n:.6g}; "
                        "set empty_cluster_strategy='reinitialize' to recover."
                    )
                warning_messages.append(
                    self._warn(
                        "Component "
                        f"{component} effective sample size {effective_n[component]:.6g} "
                        f"is below the min_effective_n heuristic {min_effective_n:.6g}; "
                        "reinitializing from a reference estimate."
                    )
                )
                if global_result is None:
                    global_result, global_warning_messages = self._build_global_reference(
                        X,
                        iteration=iteration,
                    )
                    warning_messages.extend(global_warning_messages)
                means[component] = X[rng.randint(X.shape[0])]
                covariances[component] = global_result.covariance
                precisions[component] = global_result.precision
                weights[component] = max(effective_n[component], weight_floor)
                reinitialized_components.append(component)
                glasso_n_iter.append(global_result.n_iter)
                glasso_jitter_used.append(global_result.jitter_used)
                glasso_fallback_source.append(None)
                component_update_source.append(f"reinitialized_{global_result.backend}")
                continue

            if means_override_array is None:
                means[component] = compute_weighted_mean(X, component_weights)
            else:
                means[component] = means_override_array[component]
            empirical_covariance = compute_weighted_covariance(
                X,
                component_weights,
                mean=means[component],
                reg_covar=self.reg_covar,
            )
            try:
                glasso_result = fit_graphical_lasso(
                    empirical_covariance,
                    alpha=self.alpha,
                    max_iter=self.glasso_max_iter,
                    tol=self.glasso_tol,
                    jitter=self.covariance_jitter,
                )
                covariances[component] = glasso_result.covariance
                precisions[component] = glasso_result.precision
                glasso_fallback_source.append(None)
                component_update_source.append("glasso")
            except GraphicalLassoFitError as exc:
                (
                    fallback_result,
                    fallback_source,
                    fallback_warning_messages,
                ) = self._fallback_component_estimate(
                    empirical_covariance=empirical_covariance,
                    component=component,
                    iteration=iteration,
                    previous_parameters=previous_parameters,
                    error=exc,
                )
                covariances[component] = fallback_result.covariance
                precisions[component] = fallback_result.precision
                warning_messages.extend(fallback_warning_messages)
                glasso_fallback_source.append(fallback_source)
                component_update_source.append(f"fallback_{fallback_source}")
                glasso_result = fallback_result
            weights[component] = effective_n[component]
            glasso_n_iter.append(glasso_result.n_iter)
            glasso_jitter_used.append(glasso_result.jitter_used)

        weights /= weights.sum()
        return (
            _MixtureParameters(
                weights=weights,
                means=means,
                covariances=covariances,
                precisions=precisions,
            ),
            {
                "effective_n": effective_n,
                "min_effective_n_threshold": min_effective_n,
                "near_empty_components": near_empty_components,
                "reinitialized_components": reinitialized_components,
                "glasso_n_iter": glasso_n_iter,
                "glasso_jitter_used": glasso_jitter_used,
                "glasso_fallback_source": glasso_fallback_source,
                "component_update_source": component_update_source,
                "warning_messages": warning_messages,
            },
        )

    def _effective_n_threshold(self, n_features: int) -> float:
        if self.min_effective_n is not None:
            return float(self.min_effective_n)
        return float(max(5, n_features + 1))

    def _estimate_responsibilities(
        self, X: np.ndarray, parameters: _MixtureParameters
    ) -> tuple[np.ndarray, float]:
        log_prob = self._estimate_log_component_probabilities(X, parameters)
        log_prob_norm = logsumexp(log_prob, axis=1, keepdims=True)
        responsibilities = np.exp(log_prob - log_prob_norm)
        observed_log_likelihood = float(np.sum(log_prob_norm))
        return responsibilities, observed_log_likelihood

    def _estimate_log_component_probabilities(
        self, X: np.ndarray, parameters: _MixtureParameters
    ) -> np.ndarray:
        n_features = X.shape[1]
        log_prob = np.empty((X.shape[0], self.n_components), dtype=float)
        constant = -0.5 * n_features * np.log(2.0 * np.pi)

        for component in range(self.n_components):
            diff = X - parameters.means[component]
            sign, logdet_precision = np.linalg.slogdet(parameters.precisions[component])
            if sign <= 0:
                raise ValueError("precision matrices must remain positive definite.")
            quadratic = np.einsum(
                "ni,ij,nj->n", diff, parameters.precisions[component], diff, optimize=True
            )
            log_density = constant + 0.5 * logdet_precision - 0.5 * quadratic
            weight = max(parameters.weights[component], np.finfo(float).tiny)
            log_prob[:, component] = np.log(weight) + log_density

        return log_prob

    def _compute_penalized_objective(
        self,
        *,
        observed_log_likelihood: float,
        responsibilities: np.ndarray,
        precisions: np.ndarray,
    ) -> float:
        effective_n = responsibilities.sum(axis=0)
        off_diagonal_penalty = 0.0
        for component in range(self.n_components):
            precision = precisions[component]
            mask = ~np.eye(precision.shape[0], dtype=bool)
            off_diagonal_penalty += effective_n[component] * np.sum(np.abs(precision[mask]))
        return float(observed_log_likelihood - self.alpha * off_diagonal_penalty)

    def _fitted_parameters(self) -> _MixtureParameters:
        return _MixtureParameters(
            weights=self.weights_,
            means=self.means_,
            covariances=self.covariances_,
            precisions=self.precisions_,
        )

    def _build_global_reference(
        self, X: np.ndarray, *, iteration: int
    ) -> tuple[GraphicalLassoResult, list[str]]:
        warning_messages: list[str] = []
        global_covariance = compute_weighted_covariance(
            X, np.ones(X.shape[0], dtype=float), reg_covar=self.reg_covar
        )
        try:
            result = fit_graphical_lasso(
                global_covariance,
                alpha=self.alpha,
                max_iter=self.glasso_max_iter,
                tol=self.glasso_tol,
                jitter=self.covariance_jitter,
            )
            return result, warning_messages
        except GraphicalLassoFitError:
            fallback_result = self._empirical_inverse_result(global_covariance)
            warning_messages.append(
                self._warn(
                    "Global reference graphical-lasso fit failed during "
                    f"iteration {iteration}; falling back to a stabilized "
                    "empirical covariance inverse."
                )
            )
            return fallback_result, warning_messages

    def _fallback_component_estimate(
        self,
        *,
        empirical_covariance: np.ndarray,
        component: int,
        iteration: int,
        previous_parameters: _MixtureParameters | None,
        error: GraphicalLassoFitError,
    ) -> tuple[GraphicalLassoResult, str, list[str]]:
        warning_messages: list[str] = []
        if previous_parameters is not None:
            warning_messages.append(
                self._warn(
                    "Graphical-lasso fit failed for component "
                    f"{component} during iteration {iteration}; reusing the "
                    "previous component estimate."
                )
            )
            return (
                GraphicalLassoResult(
                    covariance=previous_parameters.covariances[component].copy(),
                    precision=previous_parameters.precisions[component].copy(),
                    alpha=float(self.alpha),
                    n_iter=None,
                    costs=None,
                    backend="previous_estimate",
                    converged=None,
                    jitter_used=0.0,
                    jitter_attempted=False,
                ),
                "previous_estimate",
                warning_messages,
            )

        try:
            fallback_result = self._empirical_inverse_result(empirical_covariance)
        except GraphicalLassoFitError as fallback_error:
            raise GraphicalLassoFitError(
                "Graphical-lasso fit failed for component "
                f"{component} during iteration {iteration}, and no stable "
                "fallback estimate was available."
            ) from fallback_error
        warning_messages.append(
            self._warn(
                "Graphical-lasso fit failed for component "
                f"{component} during iteration {iteration}; falling back to a "
                "stabilized empirical covariance inverse."
            )
        )
        return fallback_result, "empirical_inverse", warning_messages

    def _empirical_inverse_result(self, covariance: np.ndarray) -> GraphicalLassoResult:
        stabilized_covariance, jitter_used = self._stabilize_covariance(covariance)
        precision = symmetrize_matrix(np.linalg.inv(stabilized_covariance))
        if not is_spd(precision):
            raise GraphicalLassoFitError(
                "Stabilized empirical covariance inverse was not positive definite."
            )
        return GraphicalLassoResult(
            covariance=stabilized_covariance,
            precision=precision,
            alpha=float(self.alpha),
            n_iter=None,
            costs=None,
            backend="empirical_inverse",
            converged=None,
            jitter_used=jitter_used,
            jitter_attempted=jitter_used > 0.0,
        )

    def _stabilize_covariance(self, covariance: np.ndarray) -> tuple[np.ndarray, float]:
        candidate = symmetrize_matrix(covariance)
        if is_spd(candidate):
            return candidate, 0.0

        jitter_value = max(self.covariance_jitter, self.reg_covar, 1e-8)
        for _ in range(8):
            stabilized = add_diagonal_jitter(candidate, jitter_value)
            if is_spd(stabilized):
                return stabilized, jitter_value
            jitter_value *= 10.0

        raise GraphicalLassoFitError("Could not stabilize covariance with diagonal jitter.")

    def _warn(self, message: str) -> str:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return message
