from typing import Callable, Optional

import numpy as np
import scipy as sp
import torch
from scipy.optimize import approx_fprime

from Distributions import TargetDistribution


class GradientCalculator:
    """Utilities for computing gradients of log target densities"""

    def __init__(self, target: TargetDistribution, epsilon: float = 1e-8):
        """
        Initialize gradient calculator.

        Args:
            target: Target distribution object
            epsilon: Step size for finite difference approximation
        """
        self.target = target
        self.epsilon = epsilon

    def log_target_density(self, x: np.ndarray) -> np.float64:
        """Compute log target density (log prior + log likelihood)"""
        return self.target.log_prior(x) + self.target.log_likelihood(x)

    def numerical_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient using finite differences.
        Uses scipy.optimize.approx_fprime for accurate approximation.
        """
        return approx_fprime(x, self.log_target_density, self.epsilon)

    def numerical_gradient_custom(self, x: np.ndarray) -> np.ndarray:
        """
        Alternative implementation using central differences.
        More accurate but slower than forward differences.
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            h = np.zeros_like(x)
            h[i] = self.epsilon
            grad[i] = (
                self.log_target_density(x + h) - self.log_target_density(x - h)
            ) / (2 * self.epsilon)
        return grad

    @staticmethod
    def create_scipy_gradient(dist: sp.stats.rv_continuous) -> Callable:
        """
        Create gradient function for scipy distributions that have
        built-in score functions.

        Args:
            dist: scipy.stats distribution object

        Returns:
            Gradient function for the log pdf
        """
        if hasattr(dist, "score"):
            return lambda x: dist.score(x)
        else:
            raise NotImplementedError(
                f"Distribution {type(dist)} does not have a score function"
            )


class MALAGradient:
    def __init__(self, target: TargetDistribution):
        self.target = target

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient using vectorized central differences"""
        epsilon = 1e-8
        x = x.astype(np.float64)
        dim = len(x)

        # Create perturbation matrix
        h_mat = np.eye(dim) * epsilon

        # Compute forward differences vectorized
        f_plus = np.array(
            [
                self.target.log_prior(x + h_mat[i])
                + self.target.log_likelihood(x + h_mat[i])
                for i in range(dim)
            ]
        )

        # Compute backward differences vectorized
        f_minus = np.array(
            [
                self.target.log_prior(x - h_mat[i])
                + self.target.log_likelihood(x - h_mat[i])
                for i in range(dim)
            ]
        )

        # Central difference approximation
        grad = (f_plus - f_minus) / (2 * epsilon)
        return grad


def get_gradient_function(
    target: TargetDistribution, method: str = "auto", epsilon: float = 1e-8
) -> Callable:
    """
    Factory function to create gradient function for multidimensional distributions.
    """

    def log_target_density(x: np.ndarray) -> np.float64:
        return target.log_prior(x) + target.log_likelihood(x)

    if method == "auto":
        return MALAGradient(target)
    if method == "numerical":

        def numerical_gradient(x: np.ndarray) -> np.ndarray:
            # Compute gradient using finite differences for each dimension
            grad = np.zeros_like(x)

            # For each dimension, compute partial derivative
            for i in range(len(x)):
                h = np.zeros_like(x)
                h[i] = epsilon  # Perturbation in i-th dimension

                # Central difference approximation
                # (f(x + h) - f(x - h)) / (2h)
                grad[i] = (log_target_density(x + h) - log_target_density(x - h)) / (
                    2 * epsilon
                )

            return grad

        return numerical_gradient
    else:
        raise ValueError(f"Unknown gradient method: {method}")
