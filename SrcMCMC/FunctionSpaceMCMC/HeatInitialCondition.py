from typing import Optional, Callable, Tuple, override

import numpy as np
import scipy as sp
from scipy.sparse import diags


from Library.LibMCMC.pCN_Analysis import (
    compute_posterior_predictive, 
    posterior_predictive_analysis,
    functional_pca_analysis
)
from Library.LibMCMC.pCN import KarhunenLoeveExpansion, PCN, PCNProposal

import tqdm
from Library.LibMCMC.Distributions import Proposal, TargetDistribution
from Library.LibMCMC.MetropolisHastings import MetropolisHastings
from Library.LibMCMC.Diagnostics import MCMCDiagnostics
from Library.LibMCMC.PRNG import RNG, SEED
import time
import functools



# Add this to your main script
if __name__ == "__main__":
    # Run test before main MCMC
    # samples, proposals = test_kl_and_propose()
    # Rest of your code...
    # Example: Heat diffusion inverse problem with PCN

    # Setup grid and create target distribution
    L = 1.0  # Domain length
    nx = 600  # Number of grid points (also number of datapoints right now which needs to change)
    x_grid = np.linspace(0, L, nx)
    t_grid = sp.stats.gamma.rvs(0.5, scale=1, loc=0, size=nx)/100
#np.full(nx, 0.1)#sp.stats.uniform(loc=0, scale=0.001).rvs(size=nx)
    obs = (x_grid, t_grid)

    # Create a target distribution for function space sampling
    class HeatEquationPrior(TargetDistribution):
        def __init__(self, x_grid, observations, noise_level=0.09, variance = 0.06):
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
            self.variance = variance
            self._array_cache = {}

            # aggregate grid points with any observation points not on grid
            # self.full_grid = np.sort(np.unique(np.concatenate([x_grid, self.x_obs])))
            
            # Generate synthetic data
            # self.true_initial = 0.8*np.sin(3*np.pi*x_grid) + 0.4*np.sin(np.pi * x_grid)  # True initial condition
            self.true_initial = lambda x: np.sin(2*np.pi * x)  # True initial condition
            solution = self._solve_heat_equation_impl(self.true_initial, self.x_obs, self.t_obs)
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
            # print("residuals: ", residuals)
            noise_var = self.variance**2
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
        beta=0.3,   # PCN step size
        n_terms=500, # Number of KL terms
    )

    # Run sampler
    sampler(6000)

    # Visualize results
    import matplotlib.pyplot as plt
    grid = np.linspace(0, 1, 1000)

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
        grid, 
        target.true_initial(grid),  # Evaluate function
        "g--", 
        linewidth=2, 
        label="True Initial Condition"
    )

    # Plot posterior samples
    for i in range(min(10, sampler._index - 1000), min(50, sampler._index - 500), 10):
        sample_func = sampler.get_function(sampler._index - i)
        plt.plot(grid, sample_func(grid), "b-", alpha=0.6)

    plt.xlabel("x")
    plt.ylabel("Function Value")
    plt.title("PCN MCMC for Heat Equation Initial Condition")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # After running your MCMC sampler:

    # Run posterior predictive analysis
    posterior_predictive_analysis(
        sampler=sampler,
        target_distribution=target,
        domain_length=L,
        prediction_times=[0.01, 0.05, 0.1, 0.2],
        n_samples=500,
        burn_in=2000
    )

    # Run FPCA analysis
    eigenvalues, eigenfunctions, scores = functional_pca_analysis(
        sampler=sampler,
        target_distribution=target,
        n_components=5,
        n_samples=1000,
        burn_in=2000,
        domain_length=L
    )
    # Simple function to evaluate posterior predictive at a specific time
    def evaluate_posterior_predictive(sampler, target, time_point=0.1, n_samples=100):
        """Evaluate posterior predictive at a specified time"""
        # Prediction grid
        x_pred = np.linspace(0, target.L, 200)
        t_pred = np.full(len(x_pred), time_point)
        
        # Get samples
        burn_in = 5000
        indices = np.linspace(burn_in, sampler._index - 1, n_samples, dtype=int)
        
        # Collect predictions
        predictions = []
        for idx in tqdm.tqdm(indices):
            func = sampler.get_function(idx)
            pred = target._solve_heat_equation_impl(func, x_pred, t_pred)
            predictions.append(pred)
        
        # Statistics
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        lower = np.percentile(predictions, 2.5, axis=0)
        upper = np.percentile(predictions, 97.5, axis=0)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.fill_between(x_pred, lower, upper, color='lightblue', alpha=0.5, label='95% CI')
        plt.plot(x_pred, mean, 'b-', linewidth=2, label='Posterior Mean')
        plt.plot(x_pred, target.true_initial(x_pred), 'g--', linewidth=2, label='True Initial')
        
        # Plot observations if they're at this time
        plt.title(f'Posterior Predictive at t = {time_point}')
        plt.xlabel('x')
        plt.ylabel('Temperature')
        plt.legend()
        plt.show()
        
        return mean, lower, upper

    # Call with your existing objects
    evaluate_posterior_predictive(sampler, target)
    plt.show()

