"""
Bayesian Optimization for Hyperparameter Tuning.

Uses Gaussian Process regression with an acquisition function (Expected Improvement)
to intelligently select hyperparameters to evaluate next.
"""

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class GaussianProcess:
    """Simple Gaussian Process regressor for Bayesian Optimization."""

    def __init__(
        self,
        length_scale: float = 1.0,
        noise: float = 1e-6,
    ):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None

    def _rbf_kernel(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Radial Basis Function (RBF) kernel."""
        if X2 is None:
            X2 = X1

        # Compute squared Euclidean distances
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        sq_dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        sq_dist = np.maximum(sq_dist, 0)  # Numerical stability

        # Apply RBF kernel
        return np.exp(-0.5 * sq_dist / (self.length_scale**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """Fit the Gaussian Process to data."""
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Compute kernel matrix
        self.K = self._rbf_kernel(X) + self.noise * np.eye(len(X))

        # Compute inverse using Cholesky decomposition for stability
        try:
            L = np.linalg.cholesky(self.K)
            self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
        except np.linalg.LinAlgError:
            # Fallback to direct inversion if Cholesky fails
            self.K_inv = np.linalg.inv(self.K + 1e-6 * np.eye(len(X)))

        self.alpha = np.dot(self.K_inv, y)
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict mean (and optionally std) at given points."""
        if self.X_train is None or self.alpha is None:
            # No data yet - return prior
            if return_std:
                return np.zeros(len(X)), np.ones(len(X))
            return np.zeros(len(X))

        K_star = self._rbf_kernel(self.X_train, X)
        K_star_star = self._rbf_kernel(X)

        mean = np.dot(K_star.T, self.alpha)

        if return_std:
            # Compute variance
            v = np.linalg.solve(
                np.linalg.cholesky(self.K), K_star
            )
            var = np.diag(K_star_star) - np.sum(v**2, axis=0)
            std = np.sqrt(np.maximum(var, 1e-10))  # Numerical stability
            return mean, std

        return mean


class BayesianOptimization:
    """Bayesian Optimization using Gaussian Process and Expected Improvement."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        discrete_params: Optional[Dict[str, List[Any]]] = None,
        n_init: int = 5,
        length_scale: float = 1.0,
        noise: float = 1e-6,
        xi: float = 0.01,
        kappa: float = 2.576,
    ):
        """
        Initialize Bayesian Optimization.

        Args:
            param_bounds: Dict of {param_name: (min, max)} for continuous params.
            discrete_params: Dict of {param_name: [possible_values]} for discrete params.
            n_init: Number of initial random samples before GP-guided search.
            length_scale: Length scale for RBF kernel.
            noise: Noise parameter for GP.
            xi: Exploration parameter for Expected Improvement.
            kappa: Controls exploration vs exploitation (higher = more exploration).
        """
        self.param_bounds = param_bounds
        self.discrete_params = discrete_params or {}
        self.n_init = n_init
        self.xi = xi
        self.kappa = kappa

        # Initialize GP
        self.gp = GaussianProcess(length_scale=length_scale, noise=noise)

        # Store evaluated points
        self.X_evaluated: List[np.ndarray] = []
        self.y_evaluated: List[float] = []

        # Track best result
        self.best_x: Optional[np.ndarray] = None
        self.best_y: float = -np.inf

        # Parameter names (ordered)
        self.param_names = list(param_bounds.keys()) + list(discrete_params.keys())

    def _sample_random(self) -> np.ndarray:
        """Sample a random point from the parameter space."""
        x = np.zeros(len(self.param_names))

        for i, name in enumerate(self.param_names):
            if name in self.param_bounds:
                # Continuous parameter - sample uniformly
                low, high = self.param_bounds[name]
                x[i] = random.uniform(low, high)
            elif name in self.discrete_params:
                # Discrete parameter - sample index from list
                values = self.discrete_params[name]
                x[i] = float(random.randint(0, len(values) - 1))

        return x

    def _params_to_indices(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert actual parameter values to indices for internal storage."""
        x = np.zeros(len(self.param_names))
        for i, name in enumerate(self.param_names):
            if name in self.param_bounds:
                x[i] = params[name]
            elif name in self.discrete_params:
                values = self.discrete_params[name]
                x[i] = float(values.index(params[name]))
        return x

    def _indices_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert internal index representation to actual parameter values."""
        result = {}
        for i, name in enumerate(self.param_names):
            if name in self.discrete_params:
                values = self.discrete_params[name]
                idx = int(round(x[i]))
                idx = max(0, min(idx, len(values) - 1))
                result[name] = values[idx]
            else:
                result[name] = x[i]
        return result

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Round discrete parameters to nearest valid index."""
        x_discrete = x.copy()

        idx = 0
        for name in self.param_names:
            if name in self.discrete_params:
                values = self.discrete_params[name]
                # Clamp index to valid range
                idx_int = int(round(x_discrete[idx]))
                idx_int = max(0, min(idx_int, len(values) - 1))
                x_discrete[idx] = float(idx_int)
            idx += 1

        return x_discrete

    def _expected_improvement(
        self,
        x: np.ndarray,
        mu: float,
        sigma: float,
    ) -> float:
        """
        Compute Expected Improvement acquisition function.

        Args:
            x: Point to evaluate (not used directly, but kept for API consistency).
            mu: Predicted mean at x.
            sigma: Predicted std at x.

        Returns:
            Expected improvement value.
        """
        if sigma <= 0:
            return 0.0

        # Improvement over current best
        improvement = self.best_y - self.xi

        # Standardize
        z = (mu - improvement) / sigma

        # Expected improvement formula
        ei = (mu - improvement) * self._norm_cdf(z) + sigma * self._norm_pdf(z)

        # Add UCB component for better exploration
        ucb = mu + self.kappa * sigma

        # Combine EI and UCB
        return ei + 0.1 * ucb

    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF using error function approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _optimize_acquisition(
        self,
        n_candidates: int = 1000,
    ) -> np.ndarray:
        """Find the point that maximizes expected improvement."""
        # Generate random candidates
        candidates = [self._sample_random() for _ in range(n_candidates)]

        # Get GP predictions
        candidate_array = np.array(candidates)
        means, stds = self.gp.predict(candidate_array, return_std=True)

        # Compute acquisition for each candidate
        eis = [
            self._expected_improvement(c, m, s)
            for c, m, s in zip(candidates, means, stds)
        ]

        # Select best candidate
        best_idx = np.argmax(eis)
        best_candidate = candidates[best_idx].copy()

        # Discretize discrete parameters
        best_candidate = self._discretize(best_candidate)

        return best_candidate

    def suggest_next(self) -> Dict[str, Any]:
        """Suggest the next hyperparameter configuration to try."""
        # If we haven't evaluated enough points, sample randomly
        if len(self.X_evaluated) < self.n_init:
            x = self._sample_random()
        else:
            # Use GP to suggest next point
            x = self._optimize_acquisition()

        # Convert indices to actual parameter values
        return self._indices_to_params(x)

    def observe(
        self,
        params: Dict[str, Any],
        target: float,
    ) -> None:
        """
        Record the result of evaluating a hyperparameter configuration.

        Args:
            params: The hyperparameter configuration that was evaluated.
            target: The target value (e.g., validation accuracy) achieved.
        """
        # Convert params to indices for internal storage
        x = self._params_to_indices(params)

        self.X_evaluated.append(x)
        self.y_evaluated.append(target)

        # Update best
        if target > self.best_y:
            self.best_y = target
            self.best_x = x.copy()

        # Refit GP
        if len(self.X_evaluated) >= 2:
            X_array = np.array(self.X_evaluated)
            y_array = np.array(self.y_evaluated)
            self.gp.fit(X_array, y_array)

    def get_best(self) -> Tuple[Dict[str, Any], float]:
        """Return the best hyperparameters found and the corresponding target value."""
        if self.best_x is None:
            return {}, self.best_y

        return self._indices_to_params(self.best_x), self.best_y

    def get_history(self) -> List[Dict[str, Any]]:
        """Return history of all evaluated configurations."""
        history = []
        for i, (x, y) in enumerate(zip(self.X_evaluated, self.y_evaluated)):
            entry = {"iteration": i + 1, "target": y}
            entry.update(self._indices_to_params(x))
            history.append(entry)
        return history


def get_bayesian_optimizer_for_model(
    model_name: str,
) -> BayesianOptimization:
    """
    Get a pre-configured BayesianOptimization instance for a specific model.

    Args:
        model_name: Name of the model architecture.

    Returns:
        Configured BayesianOptimization instance.
    """
    # Common continuous parameters
    common_bounds = {
        "lr": (-4, -2),  # Log scale: 10^lr
        "weight_decay": (-5, -3),  # Log scale: 10^wd
        "dropout": (0.1, 0.8),
    }

    # Common discrete parameters
    common_discrete = {
        "hidden_channels": [32, 64, 128, 256, 512],
        "num_layers": [2, 3, 4, 5],
        "norm": ["layer", "batch", "graph", "none"],
    }

    if model_name == "gcn":
        param_bounds = {**common_bounds}
        discrete_params = {**common_discrete}

    elif model_name == "gat":
        param_bounds = {**common_bounds}
        discrete_params = {
            **common_discrete,
            "gat_heads": [1, 2, 4, 8],
        }

    elif model_name == "sage":
        param_bounds = {**common_bounds}
        discrete_params = {**common_discrete}

    elif model_name == "ppnp":
        param_bounds = {
            **common_bounds,
            "alpha": (0.05, 0.3),
        }
        discrete_params = {**common_discrete}

    elif model_name == "appnp":
        param_bounds = {
            **common_bounds,
            "alpha": (0.05, 0.3),
        }
        discrete_params = {
            **common_discrete,
            "K": [5, 10, 15, 20, 30],
        }

    elif model_name in ("residual_gcn", "residual_gat", "residual_sage", "residual_appnp"):
        # Same as base models
        base_model = model_name.replace("residual_", "")
        return get_bayesian_optimizer_for_model(base_model)

    else:
        # Default fallback
        param_bounds = {**common_bounds}
        discrete_params = {**common_discrete}

    return BayesianOptimization(
        param_bounds=param_bounds,
        discrete_params=discrete_params,
        n_init=5,
        length_scale=1.0,
        noise=1e-6,
        xi=0.01,
        kappa=2.576,
    )


def convert_bayesian_params_to_trainable(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert Bayesian optimization output to trainable hyperparameters.

    Handles log-scale parameters and ensures all values are in valid ranges.

    Args:
        params: Raw parameters from BayesianOptimization.

    Returns:
        Processed parameters ready for training.
    """
    trainable = params.copy()

    # Convert log-scale parameters back to linear scale
    if "lr" in trainable and isinstance(trainable["lr"], (int, float)):
        trainable["lr"] = 10 ** trainable["lr"]

    if "weight_decay" in trainable and isinstance(trainable["weight_decay"], (int, float)):
        trainable["weight_decay"] = 10 ** trainable["weight_decay"]

    return trainable
