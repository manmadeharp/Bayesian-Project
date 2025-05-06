from typing import Optional, Callable, Tuple, override
from typing import Any
from dataclasses import dataclass
from numpy.random import Generator

from typing import Optional, Callable, List
import numpy as np
from numpy.random import Generator
import numpy as np
import scipy as sp
from scipy.sparse import diags


import tqdm

from .pCN_Analysis import (compute_posterior_predictive, posterior_predictive_analysis,functional_pca_analysis)
from .Distributions import Proposal, TargetDistribution
from .MetropolisHastings import MetropolisHastings
from .Diagnostics import MCMCDiagnostics
from .PRNG import RNG, SEED


# from pCN_Analysis import (compute_posterior_predictive, posterior_predictive_analysis,functional_pca_analysis)
# from Distributions import Proposal, TargetDistribution
# from MetropolisHastings import MetropolisHastings
# from Diagnostics import MCMCDiagnostics
# from PRNG import RNG, SEED
import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Create a wrapper function that handles array inputs
def _make_hashable(arr):
    return hash(arr.tobytes())

@dataclass
class KLTerms:
    """Stores parameters and derived quantities for the Karhunen-Loève expansion."""
    n_terms: int
    domain_length: float
    alpha: float
    eigenvalues: np.ndarray
    sqrt_eigenvalues: np.ndarray
    k_values: np.ndarray
    eigenfunction_norm: float
    eigenfunction: Callable[[np.ndarray, np.ndarray], np.ndarray]

    @classmethod
    def from_parameters(cls, n_terms: int, domain_length: float, alpha: float) -> 'KLTerms':
        """Initialize KL terms from domain length, alpha, and number of terms."""
        if n_terms < 1:
            raise ValueError("n_terms must be at least 1")
        if domain_length <= 0:
            raise ValueError("domain_length must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if alpha <= 0.5:
            raise ValueError("alpha must be greater than 0.5 for the covariance operator to be trace-class")
        
        k_values = np.arange(1, n_terms + 1)
        k_pi = k_values * np.pi / domain_length
        eigenvalues = np.exp(-2 * alpha * np.log(k_pi))
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        eigenfunction_norm = np.sqrt(2 / domain_length)
        eigenfunction = lambda j, x: eigenfunction_norm * np.sin(j * np.pi * x / domain_length)
        
        return cls(
            n_terms=n_terms,
            domain_length=domain_length,
            alpha=alpha,
            eigenvalues=eigenvalues,
            sqrt_eigenvalues=sqrt_eigenvalues,
            k_values=k_values,
            eigenfunction_norm=eigenfunction_norm,
            eigenfunction=eigenfunction
        )

@dataclass
class KLSample:
    """Represents a sample from the KL expansion: u(x) = sum sqrt(μ_k) * φ_k * ψ_k(x)."""
    coefficients: np.ndarray  # φ_k ~ N(0, 1)
    function: Callable[[np.ndarray], np.ndarray]  # u(x) initial condition

class KarhunenLoeveExpansion:
    """
    Karhunen–Loève expansion for Gaussian prior N(0, (-Δ)^(-α)) on [0, L].
    """
    def __init__(self, domain_length: float = 1.0, alpha: float = 2.0, n_terms: int = 100):
        """
        Set up KL basis for N(0, (-Δ)^(-α)).

        Args:
            domain_length: Length of the domain [0, L] (default: 1.0).
            alpha: Regularity parameter for covariance operator (-Δ)^(-α) (default: 2.0).
            n_terms: Number of terms in the truncated expansion (default: 100).
        """
        self.terms = KLTerms.from_parameters(n_terms, domain_length, alpha)

    def sample_coefficients(self, rng: Optional[Generator] = None) -> np.ndarray:
        """
        Sample a coefficient vector φ ~ N(0, I).

        Args:
            rng: Random number generator. If None, uses np.random.default_rng().

        Returns:
            phi: Array of shape (n_terms,) containing standard normal coefficients.
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.standard_normal(self.terms.n_terms)

    def construct_function(self, phi: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return a callable function u(x) = sum φ_k * sqrt(μ_k) * ψ_k(x).

        Args:
            phi: Coefficient vector of shape (n_terms,).

        Returns:
            u: Callable that evaluates u(x) at input points x.
        """
        if phi.shape != (self.terms.n_terms,):
            raise ValueError(f"Coefficient vector must have shape ({self.terms.n_terms},)")
        
        def u(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            eigenfunctions = self.terms.eigenfunction(self.terms.k_values[:, None], x)
            return np.sum(self.terms.sqrt_eigenvalues[:, None] * phi[:, None] * eigenfunctions, axis=0)
        
        return u


    def evaluate_on_grid(self, phi: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        """
        Evaluate the KL function on a spatial grid.

        Args:
            phi: Coefficient vector of shape (n_terms,).
            x_grid: Array of grid points where the function is evaluated.

        Returns:
            u_grid: Array of function values u(x) at x_grid points.
        """
        if phi.shape != (self.terms.n_terms,):
            raise ValueError(f"Coefficient vector must have shape ({self.terms.n_terms},)")
        x_grid = np.asarray(x_grid)
        eigenfunctions = self.terms.eigenfunction(self.terms.k_values[:, None], x_grid)
        return np.sum(self.terms.sqrt_eigenvalues[:, None] * phi[:, None] * eigenfunctions, axis=0)

    def sample_function(self, x_grid: Optional[np.ndarray] = None, rng: Optional[Generator] = None) -> KLSample:
        """
        Sample u ~ N(0, (-Δ)^(-α)) and optionally evaluate on x_grid.

        Args:
            x_grid: Array of grid points for evaluation. If None, returns the function only.
            rng: Random number generator. If None, uses np.random.default_rng().

        Returns:
            KLSample: Object containing coefficients and the function u(x).
        """
        phi = self.sample_coefficients(rng)
        u = self.construct_function(phi)
        return KLSample(coefficients=phi, function=u)

    def compute_prior_precision(self, x_grid: np.ndarray) -> np.ndarray:
        """
        Compute the precision matrix C^{-1} = P Λ^{-1} Pᵀ on x_grid.

        Args:
            x_grid: Array of grid points where the precision matrix is evaluated.

        Returns:
            precision: Precision matrix of shape (len(x_grid), len(x_grid)).
        """
        x_grid = np.asarray(x_grid)
        k = self.terms.k_values
        P = self.terms.eigenfunction(k[:, None], x_grid)
        D = np.diag(1.0 / self.terms.eigenvalues)
        return P.T @ D @ P


@dataclass(frozen=True)
class PCNConfig:
    """
    Configuration for the PCN sampler.

    Attributes:
        target_distribution: Target distribution with log_likelihood accepting np.ndarray.
        initial_state: Initial function evaluations on x_grid.
        x_grid: Spatial grid points for function evaluation.
        domain_length: Length of the domain [0, L].
        alpha: Regularity parameter for prior N(0, (-Δ)^(-α)).
        beta: PCN step size parameter (0 < beta < 1).
        n_terms: Number of terms in truncated KL expansion.
        burn_in: Number of initial samples to discard (default: 200).
    """
    target_distribution: TargetDistribution
    x_grid: np.ndarray
    domain_length: float = 1.0
    alpha: float = 2.0
    beta: float = 0.1
    n_terms: int = 100
    burn_in: int = 200

    def __post_init__(self):
        """Validate configuration parameters."""
        if not isinstance(self.target_distribution, TargetDistribution):
            raise TypeError("target_distribution must be a TargetDistribution")
        x_grid = np.asarray(self.x_grid)
        if x_grid.ndim != 1 or len(x_grid) < 2:
            raise ValueError("x_grid must be a 1D array with at least 2 points")
        if self.domain_length <= 0:
            raise ValueError("domain_length must be positive")
        if self.alpha <= 0.5:
            raise ValueError("alpha must be greater than 0.5 for trace-class covariance")
        if not 0 < self.beta < 1:
            raise ValueError("beta must be in (0, 1)")
        if self.n_terms < 1:
            raise ValueError("n_terms must be at least 1")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        # Copy arrays for immutability
        self.x_grid[...] = np.copy(x_grid)

class PCNProposal(Proposal):
    """
    Preconditioned Crank-Nicolson proposal for function space MCMC.

    Generates proposals using pCN dynamics in the KL coefficient space, evaluated on a spatial grid.
    """
    
    def __init__(
        self,
        beta: float,
        kl_expansion: KarhunenLoeveExpansion,
        rng: Optional[Generator] = None
    ):
        """
        Initialize the PCN proposal.

        Args:
            beta: Step size parameter (0 < beta < 1).
            kl_expansion: Karhunen-Loève expansion for prior sampling.
            x_grid: Spatial grid points for function evaluation.
            rng: Random number generator. If None, uses np.random.default_rng().

        Raises:
            ValueError: If beta is not in (0, 1) or x_grid is invalid.
            TypeError: If kl_expansion is not a KarhunenLoeveExpansion.
        """
        if not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")
        
        self.beta = beta
        self.kl_expansion = kl_expansion
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_sample: Optional[KLSample] = None  # Store last proposal

    def propose(self, current: KLSample) -> KLSample:
        """
        Generate a proposal using pCN dynamics in KL coefficient space.

        Args:
            current: Current state as function evaluations on x_grid.

        Returns:
            Proposed state as function evaluations on x_grid.

        Raises:
            ValueError: If current has incorrect shape.
            RuntimeError: If projection onto KL basis fails.
        """
        phi = current.coefficients
        beta = self.beta
        d = len(phi)

        # pCN update
        new_phi = np.sqrt(1 - beta**2) * phi + beta * self.rng.standard_normal(d)
        new_u = self.kl_expansion.construct_function(new_phi)

        return KLSample(coefficients=new_phi, function=new_u)

    def proposal_log_density(self, proposed: np.ndarray, current: np.ndarray) -> np.float64:
        """
        Compute the log density ratio for the pCN proposal.

        Since pCN is symmetric with respect to the prior measure, the log density ratio is zero.

        Args:
            proposed: Proposed state as function evaluations.
            current: Current state as function evaluations.

        Returns:
            Log density ratio (always 0.0).
        """
        return np.float64(0.0)

class PCN(MetropolisHastings):
    """
    Preconditioned Crank-Nicolson MCMC sampler for function spaces.

    Samples from a posterior distribution over functions using a Gaussian prior
    N(0, (-Δ)^(-α)) defined via Karhunen-Loève expansion.
    """
    
    def __init__(self, config: PCNConfig):
        """
        Initialize the PCN sampler.
        """
        super().__init__()
        self.kl_expansion = KarhunenLoeveExpansion(
            domain_length=config.domain_length,
            alpha=config.alpha,
            n_terms=config.n_terms
        )

        rng = np.random.default_rng(seed=SEED)
        phi_0 = self.kl_expansion.sample_coefficients(rng)
        u_0 = self.kl_expansion.construct_function(phi_0)
        initial_sample = KLSample(coefficients=phi_0, function=u_0)

        self.proposal_distribution = PCNProposal(
            beta=config.beta,
            kl_expansion=self.kl_expansion,
            rng=rng
        )

        self.target_distribution = config.target_distribution
        self.uniform_rng = np.random.default_rng(seed=SEED // 2)

        self.chain: List[KLSample] = [initial_sample]
        self.acceptance_rates: List[float] = [0.0]
        self._index = 1
        self.acceptance_count = 0
        self.burn_in = config.burn_in
        self.x_grid = config.x_grid.copy()

    
    def get_function(self, index: int) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return the function at a given chain index.

        Args:
            index: Index in the MCMC chain.

        Returns:
            Callable function u(x) at this index.

        Raises:
            ValueError: If index is out of bounds.
        """
        if not 0 <= index < self._index:
            raise ValueError(f"Index {index} out of bounds (0 to {self._index - 1})")
        return self.chain[index].function

    def get_functions(self, indices: List[int]) -> List[Callable[[np.ndarray], np.ndarray]]:
        """
        Return functions at multiple chain indices.

        Args:
            indices: List of chain indices.

        Returns:
            List of callable functions.

        Raises:
            ValueError: If any index is out of bounds.
        """
        return [self.get_function(idx) for idx in indices]

    def evaluate_function(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at a chain index on specified points.

        Args:
            index: Chain index.
            x: Points to evaluate the function at.

        Returns:
            Function values at specified points.

        Raises:
            ValueError: If index is out of bounds.
        """
        func = self.get_function(index)
        return func(x)

    def acceptance_ratio(self, current, proposed) -> np.float64:
        """
        Compute the log acceptance ratio for the pCN proposal.

        Since pCN is symmetric w.r.t. the prior, only the likelihood ratio is needed.

        Args:
            current: Current state (grid evaluations).
            proposed: Proposed state (grid evaluations).

        Returns:
            Log acceptance ratio.
        """
        likelihood_ratio = self.target_distribution.log_likelihood(proposed) - \
                          self.target_distribution.log_likelihood(current)
        
        return min(
                np.float64(0),
                likelihood_ratio
            )



    def burn(self, n: int):
        """
        Discard the first `n` samples as burn-in.

        Args:
            n: Number of initial samples to discard.

        Raises:
            ValueError: If burn-in exceeds available samples.
        """
        if n > self._index:
            raise ValueError("Burn-in exceeds the number of samples in the chain.")

        # Trim the chain and acceptance rate history
        self.chain = self.chain[n:]
        self.acceptance_rates = self.acceptance_rates[n:]

        # Reset counters relative to trimmed chain
        self._index -= n
        self.acceptance_count = int(self.acceptance_rates[-1] * self._index)

        # Mark burn-in complete
        self.burnt = True
    
    def __call__(self, n: int) -> None:
        """
        Run the PCN algorithm for n iterations.

        Updates the chain of KLSample objects, applying burn-in as specified in config.

        Args:
            n: Number of iterations.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError("Number of iterations must be non-negative")

        
        for _ in range(n):
            # if self._index == self.burn_in and not self.burnt:
            #     self.burn(self.burn_in)

            current_sample = self.chain[self._index - 1]
            proposed_sample = self.proposal_distribution.propose(current_sample)

            # Pass full KLSample objects to the log-likelihood
            log_alpha = self.acceptance_ratio(current_sample, proposed_sample)

            if np.log(self.uniform_rng.uniform()) < log_alpha:
                self.chain.append(proposed_sample)
                self.acceptance_count += 1
            else:
                self.chain.append(current_sample)

            self._index += 1
            self.acceptance_rates.append(self.acceptance_count / self._index)





# class PCNProposal(Proposal):
#     """PCN proposal using Karhunen-Loeve expansion for function spaces"""
#
#     def __init__(
#         self,
#         beta: float,
#         kl_expansion: KarhunenLoeveExpansion,
#         x_grid: np.ndarray,
#     ):
#         """
#         Initialize PCN proposal for function spaces
#
#         Args:
#             beta: Step size parameter (0 < beta < 1)
#             kl_expansion: Karhunen-Loeve expansion for sampling
#             x_grid: Spatial grid points
#         """
#         super().__init__(None, None) 
#         self.beta = beta
#         self.kl_expansion = kl_expansion
#         self.x_grid = x_grid
#         
#     @override
#     def propose(self, current):# -> np.ndarray:
#         """
#         Generate proposal using PCN dynamics with Karhunen-Loeve expansion
#         
#         Args:
#             current_func: Current function (with coefficients attribute)
#         
#         Returns:
#             New function from PCN dynamics
#         """
#         # Get coefficients from current function
#         current_coeffs = current.phi
#         
#         # Generate standard normal random variables for new function
#         if hasattr(self, 'rng'):
#             phi = self.rng.standard_normal(self.kl_expansion.n_terms)
#         else:
#             phi = np.random.standard_normal(self.kl_expansion.n_terms)
#         
#         # PCN dynamics directly on coefficients (more efficient)
#         proposal_coeffs = np.sqrt(1 - self.beta**2) * current_coeffs + self.beta * phi
#         
#         # Create function from new coefficients
#         return self.kl_expansion.create_function_from_coefficients(proposal_coeffs)
#
#     @override
#     def proposal_log_density(
#         self, proposed, current
#     ) -> np.float64:
#         """
#         Compute log density ratio for the PCN proposal
#
#         For PCN targeting the posterior w.r.t. prior as reference measure,
#         this is always 0 as proposals are symmetric.
#         """
#         return np.float64(0.0)
#
#     def propose_grid(self, current: np.ndarray) -> np.ndarray:
#         """
#         Generate proposal using PCN dynamics with Karhunen-Loeve expansion
#         x* = sqrt(1-beta^2)*x + beta*w, where w sim N(0,C) with C = (-Laplacian)^(-alpha)
#         """
#         # Use the proposal object from the parent class to generate standard normal random variables
#         # That will be fed into the KL expansion
#
#         # Generate a sample from the prior N(0,C) using K-L expansion
#         prior_sample = self.kl_expansion.sample_grid(self.x_grid)
#
#         # PCN proposal formula
#         return np.sqrt(1 - self.beta**2) * current + self.beta * prior_sample
#
# class PCN(MetropolisHastings):
#     """
#     Preconditioned Crank-Nicolson algorithm for function space sampling
#
#     Implements dimension-independent MCMC for function space problems
#     using the Karhunen-Loeve expansion for sampling from the prior measure.
#     """
#
#     def __init__(
#         self,
#         target_distribution: TargetDistribution,
#         initial_state: np.ndarray,
#         x_grid: np.ndarray,
#         domain_length: float = 1.0,
#         alpha: float = 4.0,
#         beta: float = 0.1,
#         n_terms: int = 100,
#     ):
#         """
#         Initialize PCN sampler for function space problems
#
#         Args:
#             target_distribution: Target distribution to sample from
#             initial_state: Starting point for the chain
#             x_grid: Spatial grid points
#             domain_length: Length of the domain [0,L]
#             alpha: Regularity parameter for prior N(0, (-delta)^(-alpha))
#             beta: PCN step size parameter (0 < beta < 1)
#             n_terms: Number of terms in truncated K-L expansion
#             reference_measure: If True, use the prior as reference measure
#         """
#         # Set up Karhunen-Loeve expansion
#         kl_expansion = KarhunenLoeveExpansion(
#             domain_length=domain_length, alpha=alpha, n_terms=n_terms
#         )
#
#         # Initialize PCN proposal
#         proposal = PCNProposal(beta=beta, kl_expansion=kl_expansion, x_grid=x_grid)
#
#         # Initialize the base MCMC sampler
#         super().__init__(target_distribution, proposal, initial_state)
#
#         self.coeff_chain = np.zeros((self.max_size, n_terms))
#
#         # Store PCN-specific attributes
#         self.kl_expansion = kl_expansion
#         self.x_grid = x_grid
#
#     def get_function(self, index):
#         """Get function from stored coefficients"""
#         phi = self.coeff_chain[index]
#         return self.kl_expansion.create_function_from_coefficients(phi)
#     
#     def get_functions(self, indices):
#         """
#         Reconstruct multiple functions from coefficient chain
#         
#         Args:
#             indices: List of chain indices to reconstruct
#             
#         Returns:
#             List of callable functions
#         """
#         return [self.get_function(idx) for idx in indices]
#     
#     def evaluate_function(self, index, x):
#         """
#         Directly evaluate function at specific points without full reconstruction
#         
#         Args:
#             index: Chain index to evaluate
#             x: Points to evaluate function at
#             
#         Returns:
#             Function values at specified points
#         """
#         func = self.get_function(index)
#         return func(x)
#
#     def acceptance_ratio(self, current, proposed):
#         """
#         Compute acceptance ratio for PCN proposal
#
#         If reference_measure=True, the prior contribution cancels out
#         and only the likelihood ratio remains.
#         """
#         # For PCN with reference measure, only the likelihood ratio matters
#         likelihood_ratio = self.target_distribution.log_likelihood(
#             proposed
#         ) - self.target_distribution.log_likelihood(current)
#
#         return min(np.float64(0), likelihood_ratio)
#
#     def __call__(self, n: int):
#         # Resize arrays if needed
#         if self._index + n > self.max_size:
#             new_max = max(self.max_size * 2, self._index + n)
#             # Resize regular chain (for backward compatibility)
#             new_chain = np.empty((new_max, self.chain.shape[1]))
#             new_chain[: self._index] = self.chain[: self._index]
#             self.chain = new_chain
#             
#             # Resize coefficient chain
#             new_coeff_chain = np.zeros((new_max, self.kl_expansion.n_terms))
#             new_coeff_chain[:self._index] = self.coeff_chain[:self._index]
#             self.coeff_chain = new_coeff_chain
#             
#             # Resize acceptance rates
#             new_acceptance_rates = np.empty(new_max)
#             new_acceptance_rates[: self._index] = self.acceptance_rates[: self._index]
#             self.acceptance_rates = new_acceptance_rates
#             self.max_size = new_max
#
#         for i in range(n): # Surely I need to add _index + i to access further values???
#             if i == 200 and not self.burnt:
#                 self.burn(199)
#
#             # Get current function
#             current = self.get_function(self._index - 1)
#             
#             # Generate proposal
#             proposed = self.proposal_distribution.propose(current)
#             
#             # Evaluate acceptance
#             log_alpha = self.acceptance_ratio(current, proposed)
#
#             if np.log(self.uniform_rng.uniform()) < log_alpha:
#                 # print(proposed.phi)
#                 # Store coefficients
#                 self.coeff_chain[self._index] = proposed.phi
#                 # Also store function values for compatibility
#                 # self.chain[self._index] = proposed(self.x_grid) # this might be the issue
#                 self.acceptance_count += 1
#             else:
#                 self.coeff_chain[self._index] = current.phi
#                 # self.chain[self._index] = current(self.x_grid)
#
#
#
#             self.acceptance_rates[self._index] = self.acceptance_count / self._index
#             self._index += 1
#
#
# # class KarhunenLoeveExpansion:
# #     """
# #     Karhunen-Loeve expansion for sampling functions from Gaussian measures
# #     """
# #
# #     def __init__(
# #         self, domain_length: float = 1.0, alpha: float = 4.0, n_terms: int = 500
# #     ):
# #         """
# #         Initialize Karhunen-Loeve expansion for sampling from N(0, (-Laplacian)^(-alpha))
# #
# #         Args:
# #             domain_length: Length of the domain [0,L]
# #             alpha: Regularity parameter for covariance operator (-Laplacian)^(-alpha)
# #             n_terms: Number of terms to use in the truncated expansion
# #         """
# #         self.L = domain_length
# #         self.alpha = alpha
# #         self.n_terms = n_terms
# #
# #     def create_function_from_coefficients(self, phi):
# #         """Create a KL function with specified coefficients"""
# #         k_values = np.arange(1, self.n_terms + 1)
# #         k_pi = k_values * np.pi / self.L
# #         eigenvalues = np.exp(-2 * self.alpha * np.log(k_pi))
# #         sqrt_eigenvalues = np.sqrt(eigenvalues)
# #         eigenfunctions = lambda x: np.sin(k_pi[:, None] * x)
# #         
# #         def kl_sample_function(x):
# #             x = np.asarray(x)
# #             return np.sum(kl_sample_function.eigenvalues[:, None] * kl_sample_function.phi[:, None] * kl_sample_function.eigenfunction(x), axis=0)
# #             
# #         kl_sample_function.phi = phi
# #         kl_sample_function.eigenvalues = sqrt_eigenvalues
# #         kl_sample_function.eigenfunction = eigenfunctions
# #         
# #         return kl_sample_function
# #     
# #     def sample(self, x_grid, rng=None):
# #         """
# #         Generate a sample function using the Karhunen-Loeve expansion
# #
# #         Args:
# #             x_grid: Spatial grid points where function is evaluated
# #             rng: Random number generator (if None, creates a new one)
# #
# #         Returns:
# #             Sample function evaluated at x_grid points
# #         """
# #         # Generate random coefficients
# #         if rng is None:
# #             phi = np.random.standard_normal(self.n_terms)
# #         else:
# #             phi = rng.standard_normal(self.n_terms)
# #             
# #         # Use the common function creation logic
# #         return self.create_function_from_coefficients(phi)
# #
# #
# #     def compute_prior_precision(self, x_grid):
# #         """
# #         Compute the precision matrix for the prior using vectorized operations
# #         """
# #         n = len(x_grid)
# #
# #         # Vectorized construction of eigenfunctions matrix
# #         # Create meshgrid of indices and positions
# #         i_indices = np.arange(1, n + 1).reshape(
# #             -1, 1
# #         )  # Column vector of indices 1,...,n
# #         x_positions = x_grid.reshape(1, -1)  # Row vector of x positions
# #
# #         # Compute all sine values at once
# #         argument = (i_indices * np.pi * x_positions) / self.L
# #         P = np.sqrt(2 / self.L) * np.sin(argument)
# #
# #         # Vectorized eigenvalues computation
# #         eigenvalues = ((np.arange(1, n+1) * np.pi / self.L)**(self.alpha))
# #         # eigenvalues = (np.arange(1, n + 1) * np.pi / self.L) ** (-self.alpha)
# #         D = np.diag(eigenvalues)
# #
# #         # Compute precision matrix
# #         precision = P @ D @ P.T
# #
# #         # Ensure symmetry
# #         # precision = (precision + precision.T) / 2
# #
# #         return precision
# #     def sample_grid(self, x_grid: np.ndarray, rng=None) -> np.ndarray:
# #         """
# #         Generate a sample function using the Karhunen-Loeve expansion
# #
# #         Args:
# #             x_grid: Spatial grid points where function is evaluated
# #             rng: Random number generator (if None, creates a new one)
# #
# #         Returns:
# #             Sample function evaluated at x_grid points
# #         """
# #         # Generate standard normal random variables
# #         if rng is None:
# #             phi = np.random.standard_normal(self.n_terms)
# #         else:
# #             phi = rng.standard_normal(self.n_terms)
# #
# #         # Pre-compute k*pi values for all terms at once
# #         k_values = np.arange(1, self.n_terms + 1)
# #         k_pi = k_values * np.pi / self.L
# #
# #         # Compute eigenvalues for all k at once
# #         eigenvalues = np.exp(-self.alpha * np.log(k_pi))     
# #         # eigenvalues = np.power(k_pi, -self.alpha)
# #
# #         # Compute eigenfunctions for all x and all k at once using broadcasting
# #         # This creates a matrix of shape (n_terms, len(x_grid))
# #         eigenfunctions = np.sin(k_pi[:, None] * x_grid)
# #
# #         # Multiply each eigenfunction by its coefficient and sum
# #         # We use broadcasting to align dimensions properly
# #         scaled_eigenfunctions = (
# #             np.sqrt(eigenvalues)[:, None] * eigenfunctions * phi[:, None]
# #         )
# #
# #         # Sum along the k-axis (axis 0) to get the final function values
# #         return np.sum(scaled_eigenfunctions, axis=0)
#
