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
        eigenvalues = np.exp(-alpha * np.log(k_pi))
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
        # super(target_distribution = config.target_distribution).__init__()
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

    def compute_posterior_mean(self, x_grid: np.ndarray = None) -> np.ndarray:
        """
        Compute the empirical posterior mean function on a grid.
        
        Args:
            x_grid: Grid points for evaluation. Uses self.x_grid if None.
            
        Returns:
            Posterior mean function evaluated on x_grid.
        """
        if x_grid is None:
            x_grid = self.x_grid
            
        # Evaluate all functions on the grid and average
        function_values = np.array([s.function(x_grid) for s in self.chain])
        return np.mean(function_values, axis=0)

    def compute_posterior_variance(self, x_grid: np.ndarray = None) -> np.ndarray:
        """
        Compute the pointwise posterior variance on a grid.
        
        Args:
            x_grid: Grid points for evaluation. Uses self.x_grid if None.
            
        Returns:
            Posterior variance at each grid point.
        """
        if x_grid is None:
            x_grid = self.x_grid
            
        function_values = np.array([s.function(x_grid) for s in self.chain])
        posterior_mean = np.mean(function_values, axis=0)
        
        # Calculate variance at each point
        return np.mean((function_values - posterior_mean)**2, axis=0)

    def compute_credible_intervals(self, x_grid: np.ndarray = None, 
                                confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pointwise credible intervals on a grid.
        
        Args:
            x_grid: Grid points for evaluation. Uses self.x_grid if None.
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound) for the credible interval at each grid point.
        """
        if x_grid is None:
            x_grid = self.x_grid
            
        posterior_mean = self.compute_posterior_mean(x_grid)
        posterior_std = np.sqrt(self.compute_posterior_variance(x_grid))
        
        # Calculate z-score for the given confidence level
        z_score = sp.stats.norm.ppf((1 + confidence) / 2)
        
        lower_bound = posterior_mean - z_score * posterior_std
        upper_bound = posterior_mean + z_score * posterior_std
        
        return lower_bound, upper_bound

    def compute_posterior_covariance(self, x_grid: np.ndarray = None) -> np.ndarray:
        """
        Compute the posterior covariance matrix on a grid.
        
        Args:
            x_grid: Grid points for evaluation. Uses self.x_grid if None.
            
        Returns:
            Posterior covariance matrix of shape (len(x_grid), len(x_grid)).
        """
        if x_grid is None:
            x_grid = self.x_grid
            
        function_values = np.array([s.function(x_grid) for s in self.chain])
        posterior_mean = np.mean(function_values, axis=0)
        
        # Calculate covariance matrix
        n_samples = len(function_values)
        centered = function_values - posterior_mean
        cov_matrix = np.zeros((len(x_grid), len(x_grid)))
        
        for i in range(n_samples):
            cov_matrix += np.outer(centered[i], centered[i])
        
        return cov_matrix / n_samples

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





