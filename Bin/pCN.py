from typing import Optional, Callable, Tuple, override

import numpy as np
import scipy as sp
from scipy.sparse import diags

from .Distributions import Proposal, TargetDistribution
from .MetropolisHastings import MetropolisHastings
from .Diagnostics import MCMCDiagnostics
from .PRNG import RNG, SEED
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

class KarhunenLoeveExpansion:
    """
    Karhunen-Loeve expansion for sampling functions from Gaussian measures
    """

    def __init__(
        self, domain_length: float = 1.0, alpha: float = 4.0, n_terms: int = 500
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

    def create_function_from_coefficients(self, phi):
        """Create a KL function with specified coefficients"""
        k_values = np.arange(1, self.n_terms + 1)
        k_pi = k_values * np.pi / self.L
        eigenvalues = np.exp(-self.alpha * np.log(k_pi))
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        eigenfunctions = lambda x: np.sin(k_pi[:, None] * x)
        
        def kl_sample_function(x):
            x = np.asarray(x)
            return np.sum(kl_sample_function.eigenvalues[:, None] * kl_sample_function.phi[:, None] * kl_sample_function.eigenfunction(x), axis=0)
            
        kl_sample_function.phi = phi
        kl_sample_function.eigenvalues = sqrt_eigenvalues
        kl_sample_function.eigenfunction = eigenfunctions
        
        return kl_sample_function
    
    def sample(self, x_grid, rng=None):
        """
        Generate a sample function using the Karhunen-Loeve expansion

        Args:
            x_grid: Spatial grid points where function is evaluated
            rng: Random number generator (if None, creates a new one)

        Returns:
            Sample function evaluated at x_grid points
        """
        # Generate random coefficients
        if rng is None:
            phi = np.random.standard_normal(self.n_terms)
        else:
            phi = rng.standard_normal(self.n_terms)
            
        # Use the common function creation logic
        return self.create_function_from_coefficients(phi)


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
        eigenvalues = ((np.arange(1, n+1) * np.pi / self.L)**(2*self.alpha))
        # eigenvalues = (np.arange(1, n + 1) * np.pi / self.L) ** (-self.alpha)
        D = np.diag(eigenvalues)

        # Compute precision matrix
        precision = P @ D @ P.T

        # Ensure symmetry
        # precision = (precision + precision.T) / 2

        return precision
    def sample_grid(self, x_grid: np.ndarray, rng=None) -> np.ndarray:
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
        eigenvalues = np.exp(-self.alpha * np.log(k_pi))     
        # eigenvalues = np.power(k_pi, -self.alpha)

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
        super().__init__(None, None) 
        self.beta = beta
        self.kl_expansion = kl_expansion
        self.x_grid = x_grid
        
    @override
    def propose(self, current):# -> np.ndarray:
        """
        Generate proposal using PCN dynamics with Karhunen-Loeve expansion
        
        Args:
            current_func: Current function (with coefficients attribute)
        
        Returns:
            New function from PCN dynamics
        """
        # Get coefficients from current function
        current_coeffs = current.phi
        
        # Generate standard normal random variables for new function
        if hasattr(self, 'rng'):
            phi = self.rng.standard_normal(self.kl_expansion.n_terms)
        else:
            phi = np.random.standard_normal(self.kl_expansion.n_terms)
        
        # PCN dynamics directly on coefficients (more efficient)
        proposal_coeffs = np.sqrt(1 - self.beta**2) * current_coeffs + self.beta * phi
        
        # Create function from new coefficients
        return self.kl_expansion.create_function_from_coefficients(proposal_coeffs)

    @override
    def proposal_log_density(
        self, proposed, current
    ) -> np.float64:
        """
        Compute log density ratio for the PCN proposal

        For PCN targeting the posterior w.r.t. prior as reference measure,
        this is always 0 as proposals are symmetric.
        """
        return np.float64(0.0)
    @override
    def propose_grid(self, current: np.ndarray) -> np.ndarray:
        """
        Generate proposal using PCN dynamics with Karhunen-Loeve expansion
        x* = sqrt(1-beta^2)*x + beta*w, where w sim N(0,C) with C = (-Laplacian)^(-alpha)
        """
        # Use the proposal object from the parent class to generate standard normal random variables
        # That will be fed into the KL expansion

        # Generate a sample from the prior N(0,C) using K-L expansion
        prior_sample = self.kl_expansion.sample_grid(self.x_grid)

        # PCN proposal formula
        return np.sqrt(1 - self.beta**2) * current + self.beta * prior_sample

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
        alpha: float = 4.0,
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

        self.coeff_chain = np.zeros((self.max_size, n_terms))

        # Store PCN-specific attributes
        self.kl_expansion = kl_expansion
        self.x_grid = x_grid

    def get_function(self, index):
        """Get function from stored coefficients"""
        phi = self.coeff_chain[index]
        return self.kl_expansion.create_function_from_coefficients(phi)
    
    def get_functions(self, indices):
        """
        Reconstruct multiple functions from coefficient chain
        
        Args:
            indices: List of chain indices to reconstruct
            
        Returns:
            List of callable functions
        """
        return [self.get_function(idx) for idx in indices]
    
    def evaluate_function(self, index, x):
        """
        Directly evaluate function at specific points without full reconstruction
        
        Args:
            index: Chain index to evaluate
            x: Points to evaluate function at
            
        Returns:
            Function values at specified points
        """
        func = self.get_function(index)
        return func(x)

    def acceptance_ratio(self, current, proposed):
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
        # Resize arrays if needed
        if self._index + n > self.max_size:
            new_max = max(self.max_size * 2, self._index + n)
            # Resize regular chain (for backward compatibility)
            new_chain = np.empty((new_max, self.chain.shape[1]))
            new_chain[: self._index] = self.chain[: self._index]
            self.chain = new_chain
            
            # Resize coefficient chain
            new_coeff_chain = np.zeros((new_max, self.kl_expansion.n_terms))
            new_coeff_chain[:self._index] = self.coeff_chain[:self._index]
            self.coeff_chain = new_coeff_chain
            
            # Resize acceptance rates
            new_acceptance_rates = np.empty(new_max)
            new_acceptance_rates[: self._index] = self.acceptance_rates[: self._index]
            self.acceptance_rates = new_acceptance_rates
            self.max_size = new_max

        for i in range(n): # Surely I need to add _index + i to access further values???
            if i == 200 and not self.burnt:
                self.burn(199)

            # Get current function
            current = self.get_function(self._index - 1)
            
            # Generate proposal
            proposed = self.proposal_distribution.propose(current)
            
            # Evaluate acceptance
            log_alpha = self.acceptance_ratio(current, proposed)

            if np.log(self.uniform_rng.uniform()) < log_alpha:
                # print(proposed.phi)
                # Store coefficients
                self.coeff_chain[self._index] = proposed.phi
                # Also store function values for compatibility
                # self.chain[self._index] = proposed(self.x_grid) # this might be the issue
                self.acceptance_count += 1
            else:
                self.coeff_chain[self._index] = current.phi
                # self.chain[self._index] = current(self.x_grid)



            self.acceptance_rates[self._index] = self.acceptance_count / self._index
            self._index += 1


def test_kl_and_propose():
    """Test KL expansion and proposal mechanism"""
    import matplotlib.pyplot as plt
    
    # Setup
    L = 1.0
    x_grid = np.linspace(0, L, 1000)
    alpha = 4.0
    beta = 0.2
    n_terms = 500
    
    # Create a KL expansion object
    kl = KarhunenLoeveExpansion(domain_length=L, alpha=alpha, n_terms=n_terms)
    
    # Create a known function
    true_func = lambda x: np.sin(np.pi * x / L)
    
    # Sample from KL
    samples = []
    for i in range(100):
        sample_func = kl.sample(x_grid)
        samples.append(sample_func)

    # Sample from KL
    samples2 = []
    for i in range(100):
        sample_func = kl.sample_grid(x_grid)
        samples2.append(sample_func)
    
    # Create proposal object
    proposal = PCNProposal(beta=beta, kl_expansion=kl, x_grid=x_grid)
    
    # Test proposals
    proposals = []
    current = kl.create_function_from_coefficients(np.zeros(n_terms))  # Start with zero
    for i in range(5):
        proposed = proposal.propose(current)
        proposals.append(proposed)
        current = proposed
    proposals2 = []
    current = samples2[1]
    for i in range(5):
        proposed = proposal.propose_grid(current)
        proposals2.append(proposed)
        current = proposed
    
    # Plot everything
    plt.figure(figsize=(12, 8))
    
    # Plot true function
    plt.plot(x_grid, true_func(x_grid), 'r-', label='True Function')
    
    # Plot KL samples
    for i, sample in enumerate(samples):
        plt.plot(x_grid, sample(x_grid), 'b-', alpha=0.3, label=f'KL Sample {i+1}' if i==0 else None)
    

    # Plot proposals
    for i, prop in enumerate(proposals):
        plt.plot(x_grid, prop(x_grid), 'g-', alpha=0.3, label=f'Proposal {i+1}' if i==0 else None)

    plt.legend()
    plt.title('KL Samples and PCN Proposals')
    plt.grid(True)
    plt.show()

    plt.plot(x_grid, true_func(x_grid), 'r-', label='True Function')
    for i, sample in enumerate(samples2):
        plt.plot(x_grid, sample, 'o-', alpha=0.3, label=f'KL Sample {i+1}' if i==0 else None)
    for i, prop in enumerate(proposals2):
        plt.plot(x_grid, prop, 'g-', alpha=0.3, label=f'Proposal {i+1}' if i==0 else None)
    
    plt.legend()
    plt.title('KL Samples and PCN Proposals')
    plt.grid(True)
    plt.show()
    
    return samples, proposals

# Add this to your main script
if __name__ == "__main__":
    # Run test before main MCMC
    samples, proposals = test_kl_and_propose()
    # Rest of your code...
    # Example: Heat diffusion inverse problem with PCN

    # Setup grid and create target distribution
    L = 1.0  # Domain length
    nx = 5000  # Number of grid points (also number of datapoints right now which needs to change)
    x_grid = np.linspace(0, L, nx)
    t_grid = np.full(5000, 0.01)#sp.stats.uniform(loc=0, scale=0.001).rvs(size=nx)
    obs = (x_grid, t_grid)

    # Create a target distribution for function space sampling
    class HeatEquationPrior(TargetDistribution):
        def __init__(self, x_grid, observations, noise_level=0.01):
            """
            Create a target for the heat equation initial condition
            """
            # Store grid information
            self.x_grid = x_grid
            self.L = x_grid[-1] - x_grid[0]  # Domain length
            self.n_terms = 400  # Number of terms in Fourier series

            self.observations = observations  # Observation time
            self.x_obs, self.t_obs = self.observations
            self.noise_level = noise_level
            self._array_cache = {}

            # aggregate grid points with any observation points not on grid
            # self.full_grid = np.sort(np.unique(np.concatenate([x_grid, self.x_obs])))
            
            # Generate synthetic data
            # self.true_initial = 0.8*np.sin(3*np.pi*x_grid) + 0.4*np.sin(np.pi * x_grid)  # True initial condition
            self.true_initial = lambda x: np.sin(np.pi * x)  # True initial condition
            solution = self.solve_heat_equation(self.true_initial, self.x_obs, self.t_obs)
            self.data = solution + self.noise_level * np.random.randn(len(self.x_obs))
            
            # Set up KL expansion for the prior
            self.alpha = 4
            self.kl = KarhunenLoeveExpansion(
                domain_length=self.L,
                alpha=self.alpha
            )
            # Initialize with placeholder values (not actually used)
            super().__init__(None, None, self.data, noise_level)

            # Compute prior precision matrix
            self.prior_precision = self.kl.compute_prior_precision(self.x_obs)

        def solve_heat_equation(self, initial_func, x, t):
            """Heat equation solver with simple caching"""
            # Create a unique key for this calculation
            if hasattr(initial_func, 'phi'):
                # For KL functions, use their identity
                cache_key = id(initial_func)
                
                # Check if we already solved for this function and grid combo
                if cache_key in self._array_cache:
                    return self._array_cache[cache_key]
                
                # Solve heat equation
                solution = self._solve_heat_equation_impl(initial_func, x, t)
                
                # Cache the result
                self._array_cache[cache_key] = solution
                
                return solution
            else:
                # For non-KL functions, just solve directly
                return self._solve_heat_equation_impl(initial_func, x, t)

        @functools.lru_cache(maxsize=1000)
        def _solve_heat_equation_cached(self, coeffs_tuple, x_tuple, t):
            """Cached version of heat equation solver"""
            # Convert back to numpy arrays
            coeffs = np.array(coeffs_tuple)
            x = np.array(x_tuple)
            
            # Recreate the function
            initial_func = self.kl.create_function_from_coefficients(coeffs)
            
            # Call the implementation
            return self._solve_heat_equation_impl(initial_func, x, t)
        def _solve_heat_equation_impl(self, initial, x, t): 
            """
            Fully vectorized heat equation solver using Fourier series for true solutions
            """
            # Generate k values
            k = np.arange(1, self.n_terms + 1)
            k_reshaped = k[:, None]  # Shape: (n_terms, 1)
            
            # Calculate basis functions (eigenfunctions)
            basis = np.sin(k_reshaped * np.pi * x / self.L)
            
            # Calculate Fourier coefficients
            # coeffs = np.zeros(self.n_terms)
            # initial is shape (nx,), basis is shape (n_terms, nx)
            # Need to reshape initial to broadcast correctly
            initial_reshaped = initial(x).reshape(1, -1)  # Shape (1, nx)
            coeffs = np.trapezoid(initial_reshaped * basis, x, axis=1)
            
            # Apply time evolution and sum
            time_factor = np.exp(-(k_reshaped * np.pi / self.L)**2 * t)

            solution = np.sum(coeffs[:, None] * time_factor * basis, axis=0)
            
            return solution
        
        def log_likelihood(self, x) -> np.float64:
            """Compute log likelihood for heat equation observations"""
            # Forward model: use the same solver as for data generation
            predicted = self._solve_heat_equation_impl(x, self.x_obs, self.t_obs)
            
            # Gaussian likelihood
            residuals = self.data - predicted
            print("residuals: ", residuals)
            noise_var = self.noise_level**2
            log_lik = -0.5 * np.sum(residuals**2) / noise_var
            log_lik -= 0.5 * len(self.data) * np.log(2 * np.pi * noise_var)

            return np.float64(log_lik)

        def log_prior(self, func):
            """Compute log prior density for KL functions"""
            # if hasattr(func, 'coefficients'):
            #     # For KL functions, use coefficients directly (more efficient)
            #     phi = func.phi
            #     return -0.5 * np.sum(phi**2) - 0.5 * len(phi) * np.log(2 * np.pi)
            # else:
                # Fallback - evaluate function and use precision matrix
            func_values = func(self.x_obs)
            return -0.5 * func_values @ self.prior_precision @ func_values - 0.5 * len(func_values) * np.log(2 * np.pi)

    # Create target and initial state
    target = HeatEquationPrior(x_grid, obs)
    initial_state = np.zeros_like(x_grid)  # Start from zeros

    # Create PCN sampler for function space
    sampler = PCN(
        target_distribution=target,
        initial_state=initial_state,
        x_grid=x_grid,
        domain_length=L,
        alpha=4,  # Prior regularity
        beta=0.2,   # PCN step size
        n_terms=500, # Number of KL terms
    )

    # Run sampler
    sampler(2000)

    # Visualize results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Plot true initial condition, data, and posterior samples
    plt.scatter(
        x_grid, target.data, color="black", alpha=0.5, label="Observations at t=0.1"
    )

    # Plot some posterior samples
    # for i in range(min(10, sampler._index - 800), min(50, sampler._index - 600), 10):
    #     plt.plot(x_grid, sampler.chain[sampler._index - i], "o-", alpha=0.6)
    #
    # Plot posterior mean
    # values = []
    # for i in range(8000, 10000-1):
    #     value = sampler.get_function(i)(x_grid)
    #     values.append(value)
    # post_mean = np.mean(values, axis=0)
    # plt.plot(x_grid, post_mean, 'r-', linewidth=2, label='Posterior Mean')

    # Plot true initial condition
    # Plot true initial condition
    plt.plot(
        x_grid, 
        target.true_initial(x_grid),  # Evaluate function
        "g--", 
        linewidth=2, 
        label="True Initial Condition"
    )

    # Plot posterior samples
    for i in range(min(10, sampler._index - 800), min(50, sampler._index - 600), 10):
        sample_func = sampler.get_function(sampler._index - i)
        plt.plot(x_grid, sample_func(x_grid), "b-", alpha=0.6)

    plt.xlabel("x")
    plt.ylabel("Function Value")
    plt.title("PCN MCMC for Heat Equation Initial Condition")
    plt.legend()
    plt.tight_layout()
    plt.show()
