from typing import Optional, Callable, Tuple

import numpy as np
import scipy as sp
from scipy.sparse import diags

from Distributions import Proposal, TargetDistribution
from MetropolisHastings import MetropolisHastings
from PRNG import RNG, SEED


class KarhunenLoeveExpansion:
    """
    Karhunen-Loeve expansion for sampling functions from Gaussian measures
    """

    def __init__(
        self, domain_length: float = 1.0, alpha: float = 2.0, n_terms: int = 100
    ):
        """
        Initialize Karhunen-Loeve expansion for sampling from N(0, (-Laplacian)^(-alpha))

        Args:
            domain_length: Length of the domain [0,L]
            alpha: Regularity parameter for covariance operator (-Laplacian)^(-alpha)
            n_terms: Number of terms to use in the truncated expansion
        """
        self.L = domain_length
        self.alpha = alpha
        self.n_terms = n_terms

    def sample(self, x_grid: np.ndarray, rng=None) -> np.ndarray:
        """
        Generate a sample function using the Karhunen-Loeve expansion

        Args:
            x_grid: Spatial grid points where function is evaluated
            rng: Random number generator (if None, creates a new one)

        Returns:
            Sample function evaluated at x_grid points
        """
        # Generate standard normal random variables
        if rng is None:
            phi = np.random.standard_normal(self.n_terms)
        else:
            phi = rng.standard_normal(self.n_terms)

        # Pre-compute k*pi values for all terms at once
        k_values = np.arange(1, self.n_terms + 1)
        k_pi = k_values * np.pi / self.L

        # Compute eigenvalues for all k at once
        eigenvalues = np.power(k_pi, -self.alpha)

        # Compute eigenfunctions for all x and all k at once using broadcasting
        # This creates a matrix of shape (n_terms, len(x_grid))
        eigenfunctions = np.sin(k_pi[:, None] * x_grid)

        # Multiply each eigenfunction by its coefficient and sum
        # We use broadcasting to align dimensions properly
        scaled_eigenfunctions = (
            np.sqrt(eigenvalues)[:, None] * eigenfunctions * phi[:, None]
        )

        # Sum along the k-axis (axis 0) to get the final function values
        return np.sum(scaled_eigenfunctions, axis=0)

    def compute_prior_precision(self, x_grid):
        """
        Compute the precision matrix for the prior using vectorized operations
        """
        n = len(x_grid)

        # Vectorized construction of eigenfunctions matrix
        # Create meshgrid of indices and positions
        i_indices = np.arange(1, n + 1).reshape(
            -1, 1
        )  # Column vector of indices 1,...,n
        x_positions = x_grid.reshape(1, -1)  # Row vector of x positions

        # Compute all sine values at once
        argument = (i_indices * np.pi * x_positions) / self.L
        P = np.sqrt(2 / self.L) * np.sin(argument)

        # Vectorized eigenvalues computation
        eigenvalues = (np.arange(1, n + 1) * np.pi / self.L) ** (-self.alpha)
        D = np.diag(eigenvalues)

        # Compute precision matrix
        precision = P @ D @ P.T

        # Ensure symmetry
        precision = (precision + precision.T) / 2

        return precision


class PCNProposal(Proposal):
    """PCN proposal using Karhunen-Loeve expansion for function spaces"""

    def __init__(
        self,
        beta: float,
        kl_expansion: KarhunenLoeveExpansion,
        x_grid: np.ndarray,
    ):
        """
        Initialize PCN proposal for function spaces

        Args:
            beta: Step size parameter (0 < beta < 1)
            kl_expansion: Karhunen-Loeve expansion for sampling
            x_grid: Spatial grid points
        """
        super().__init__(sp.stats.norm, 1.0)  # Initialize with dummy scale
        self.beta = beta
        self.kl_expansion = kl_expansion
        self.x_grid = x_grid

    def propose(self, current: np.ndarray) -> np.ndarray:
        """
        Generate proposal using PCN dynamics with Karhunen-Loeve expansion

        x* = sqrt(1-beta^2)*x + beta*w, where w sim N(0,C) with C = (-Laplacian)^(-alpha)
        """
        # Use the proposal object from the parent class to generate standard normal random variables
        # That will be fed into the KL expansion

        # Generate a sample from the prior N(0,C) using K-L expansion
        prior_sample = self.kl_expansion.sample(self.x_grid)

        # PCN proposal formula
        return np.sqrt(1 - self.beta**2) * current + self.beta * prior_sample

    def proposal_log_density(
        self, proposed: np.ndarray, current: np.ndarray
    ) -> np.float64:
        """
        Compute log density ratio for the PCN proposal

        For PCN targeting the posterior w.r.t. prior as reference measure,
        this is always 0 as proposals are symmetric.
        """
        return np.float64(0.0)


class PCN(MetropolisHastings):
    """
    Preconditioned Crank-Nicolson algorithm for function space sampling

    Implements dimension-independent MCMC for function space problems
    using the Karhunen-Loeve expansion for sampling from the prior measure.
    """

    def __init__(
        self,
        target_distribution: TargetDistribution,
        initial_state: np.ndarray,
        x_grid: np.ndarray,
        domain_length: float = 1.0,
        alpha: float = 2.0,
        beta: float = 0.1,
        n_terms: int = 100,
    ):
        """
        Initialize PCN sampler for function space problems

        Args:
            target_distribution: Target distribution to sample from
            initial_state: Starting point for the chain
            x_grid: Spatial grid points
            domain_length: Length of the domain [0,L]
            alpha: Regularity parameter for prior N(0, (-delta)^(-alpha))
            beta: PCN step size parameter (0 < beta < 1)
            n_terms: Number of terms in truncated K-L expansion
            reference_measure: If True, use the prior as reference measure
        """
        # Set up Karhunen-Loeve expansion
        kl_expansion = KarhunenLoeveExpansion(
            domain_length=domain_length, alpha=alpha, n_terms=n_terms
        )

        # Initialize PCN proposal
        proposal = PCNProposal(beta=beta, kl_expansion=kl_expansion, x_grid=x_grid)

        # Initialize the base MCMC sampler
        super().__init__(target_distribution, proposal, initial_state)

        # Store PCN-specific attributes
        self.kl_expansion = kl_expansion
        self.x_grid = x_grid

    def acceptance_ratio(self, current: np.ndarray, proposed: np.ndarray) -> np.float64:
        """
        Compute acceptance ratio for PCN proposal

        If reference_measure=True, the prior contribution cancels out
        and only the likelihood ratio remains.
        """
        # For PCN with reference measure, only the likelihood ratio matters
        likelihood_ratio = self.target_distribution.log_likelihood(
            proposed
        ) - self.target_distribution.log_likelihood(current)

        return min(np.float64(0), likelihood_ratio)

    def __call__(self, n: int):
        """Run PCN for n iterations"""
        if self._index + n > self.max_size:
            new_max = max(self.max_size * 2, self._index + n)
            new_chain = np.empty((new_max, self.chain.shape[1]))
            new_chain[: self._index] = self.chain[: self._index]
            self.chain = new_chain
            new_acceptance_rates = np.empty(new_max)
            new_acceptance_rates[: self._index] = self.acceptance_rates[: self._index]
            self.acceptance_rates = new_acceptance_rates
            self.max_size = new_max

        for i in range(n):
            if i == 200 and not self.burnt:
                self.burn(199)

            current = self.chain[self._index - 1]
            proposed = self.proposal_distribution.propose(current)

            log_alpha = self.acceptance_ratio(current, proposed)

            if np.log(self.uniform_rng.uniform()) < log_alpha:
                self.chain[self._index] = proposed
                self.acceptance_count += 1
            else:
                self.chain[self._index] = current

            self.acceptance_rates[self._index] = self.acceptance_count / self._index
            self._index += 1


if __name__ == "__main__":
    # Example: Heat diffusion inverse problem with PCN

    # Setup grid and create target distribution
    L = 1.0  # Domain length
    nx = 100  # Number of grid points
    x_grid = np.linspace(0, L, nx)

    # Create a target distribution for function space sampling
    class HeatEquationPrior(TargetDistribution):
        def __init__(self, x_grid, noise_level=0.05):
            """
            Create a target for the heat equation initial condition
            """
            # Generate synthetic data
            true_initial = np.sin(np.pi * x_grid)  # True initial condition

            # Forward model (simple for illustration)
            def solve_heat_equation(initial, t=0.1):
                # Simple soluiton
                return initial * np.exp(-(np.pi**2) * t)

            # Generate observations at t=0.1
            self.x_grid = x_grid
            self.t_obs = 0.1
            solution = solve_heat_equation(true_initial, self.t_obs)
            self.data = solution + noise_level * np.random.randn(len(x_grid))

            # Store noise level as an instance variable
            self.noise_level = noise_level

            # Set up KL expansion for the prior
            self.alpha = 2.0
            self.kl = KarhunenLoeveExpansion(domain_length=1.0, alpha=self.alpha)

            # Initialize with placeholder values (not actually used)
            super().__init__(None, None, self.data, noise_level)

            # Compute prior precision matrix
            self.prior_precision = self.kl.compute_prior_precision(x_grid)

        def log_likelihood(self, x: np.ndarray) -> np.float64:
            """Compute log likelihood for heat equation observations"""
            # Forward model: simple analytic solution
            predicted = x * np.exp(-(np.pi**2) * self.t_obs)

            # Gaussian likelihood
            residuals = self.data - predicted
            noise_var = self.noise_level**2
            log_lik = -0.5 * np.sum(residuals**2) / noise_var
            log_lik -= 0.5 * len(self.data) * np.log(2 * np.pi * noise_var)

            return np.float64(log_lik)

        def log_prior(self, x: np.ndarray) -> np.float64:
            """Compute log prior density using KL prior structure"""
            # Quadratic form for zero-mean Gaussian prior
            log_prior = -0.5 * x @ self.prior_precision @ x

            # Normalization constant (up to constant)
            log_prior -= 0.5 * len(x) * np.log(2 * np.pi)

            return np.float64(log_prior)

    # Create target and initial state
    target = HeatEquationPrior(x_grid)
    initial_state = np.zeros_like(x_grid)  # Start from zeros

    # Create PCN sampler for function space
    sampler = PCN(
        target_distribution=target,
        initial_state=initial_state,
        x_grid=x_grid,
        domain_length=L,
        alpha=4,  # Prior regularity
        beta=0.2,  # PCN step size
        n_terms=100,  # Number of KL terms
    )

    # Run sampler
    sampler(5000)

    # Visualize results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Plot true initial condition, data, and posterior samples
    plt.scatter(
        x_grid, target.data, color="black", alpha=0.5, label="Observations at t=0.1"
    )

    # Plot some posterior samples
    for i in range(min(10, sampler._index - 800), min(50, sampler._index - 100), 10):
        plt.plot(x_grid, sampler.chain[sampler._index - i], "b-", alpha=0.2)

    # Plot posterior mean
    post_mean = np.mean(
        sampler.chain[max(0, sampler._index - 500) : sampler._index], axis=0
    )
    plt.plot(x_grid, post_mean, "r-", linewidth=2, label="Posterior Mean")

    # Plot true initial condition
    plt.plot(
        x_grid,
        np.sin(np.pi * x_grid),
        "g--",
        linewidth=2,
        label="True Initial Condition",
    )

    plt.xlabel("x")
    plt.ylabel("Function Value")
    plt.title("PCN MCMC for Heat Equation Initial Condition")
    plt.legend()
    plt.tight_layout()
    plt.show()
