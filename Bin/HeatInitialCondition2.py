# from typing import Callable, Tuple, Union
# import numpy as np
# import matplotlib.pyplot as plt
# from dataclasses import dataclass, field
#
# from Library.LibMCMC.pCN import KarhunenLoeveExpansion, PCN, KLSample
# from Library.LibMCMC.Distributions import TargetDistribution
# from Library.LibMCMC.PRNG import RNG, SEED
#
#
# from Library.LibMCMC.pCN_Analysis import (
#     compute_posterior_predictive, 
#     posterior_predictive_analysis,
#     functional_pca_analysis
# )
# from Library.LibMCMC.pCN import KarhunenLoeveExpansion, PCN, PCNProposal
#
# import tqdm
# from Library.LibMCMC.Distributions import Proposal, TargetDistribution
# from Library.LibMCMC.MetropolisHastings import MetropolisHastings
# from Library.LibMCMC.Diagnostics import MCMCDiagnostics
# import time
# import functools
#
#
# # # Grid and observation setup
# # L = 1.0
# # x_grid = np.linspace(0, L, 128)
# # x_obs = np.linspace(0, L, 30)
# # t_obs = 0.1
# # obs = (x_obs, t_obs)
#
# @dataclass
# class HeatEquationTargetConfig:
#     x_grid: np.ndarray               # Grid for integration
#     x_obs: np.ndarray                # Observation locations (space)
#     t_obs: np.ndarray                # Observation times
#     true_initial: Callable[[np.ndarray], np.ndarray] = lambda x: np.sin(2 * np.pi * x)
#     noise_level: float = 0.07
#     variance: float = 0.05
#     seed: int = 42
#
#     fourier_coefficients: np.ndarray = []  # a_k = <u0, phi_k>
#     data_vector: np.ndarray  = []           # Flattened observation vector
#     L: float   = 1                         # Domain length
#     k: np.ndarray   = 200                    # Modes 1..k
#
#     alpha = 2 # this is for KL expansion
#
#     def __post_init__(self):
#         self.L = self.x_grid[-1] - self.x_grid[0]
#         self.k = np.arange(1, self.k + 1).reshape(-1, 1)
#
#
#     def create(self) -> "HeatEquationTarget":
#         target = HeatEquationTarget(self)
#         target.generate_data()  # Generate data with defaults
#         return target
#
#     def get_size(self) -> Tuple[int, int]:
#         return (len(self.x_obs), len(self.t_obs))
#
# class HeatEquationTarget:
#     def __init__(self, config: HeatEquationTargetConfig):
#         self.config = config
#         self.x_obs = config.x_obs
#         self.t_obs = config.t_obs
#         self.data_vector = config.data_vector
#         self.variance = config.variance
#         self.L = config.L
#         self.k = config.k  # (d, 1)
#         self.true_initial = config.true_initial
#
#
#         self.data_vector = None
#         self.prior_precision = self._construct_prior_precision()
#
#         # Simulate data using known true initial condition
#         # # clean = self._solve_heat_equation_impl(self.true_initial, self.x_obs, self.t_obs)
#         # rng = np.random.default_rng(seed=42)
#         # noise = self.noise_level * rng.normal(size=clean.shape)
#         # self.data = clean + noise
#         # self.data_vector = self.data.flatten()
#
#         # Define fallback prior precision matrix over observation grid
#         self.prior_precision = self._construct_prior_precision()
#
#     def solve_from_initial(self, u0: Callable[[np.ndarray], np.ndarray], x: np.ndarray, t: np.ndarray) -> np.ndarray:
#         """
#         Solve the heat equation from a callable initial condition.
#
#         Args:
#             u0: Callable u_0(x) representing the initial condition.
#             x: Array of spatial points.
#             t: Array of time points.
#
#         Returns:
#             u_xt: Solution u(x,t) with shape (len(x), len(t)).
#         """
#         # Project u0 onto normalized basis phi_k(x) = sqrt(2/L) * sin(k * pi * x / L)
#         phi_k_grid = np.sqrt(2 / self.L) * np.sin(self.k * np.pi * self.x_obs.reshape(1, -1) / self.L)  # (d, n_grid)
#         u0_vals = u0(self.x_obs).reshape(1, -1)  # (1, n_grid)
#         dx = self.x_obs[1] - self.x_obs[0]
#         coeffs = np.sum(u0_vals * phi_k_grid, axis=1) * dx  # (d,)
#         
#         # Adjust for unnormalized basis sin(k * pi * x / L)
#         coeffs *= np.sqrt(2 / self.L)
#         
#         return self.solve_from_coefficients(coeffs, x, t)
#
#     def solve_from_coefficients(
#         self,
#         coeffs: np.ndarray,
#         x: np.ndarray,
#         t: np.ndarray
#     ) -> np.ndarray:
#         """
#         Evaluate the forward model u(t,x) from Fourier coefficients over basis sin(kπx/L).
#         """
#         phi_k_x = np.sin(self.k * np.pi * x.reshape(1, -1) / self.L)  # (d, n_x)
#         decay = np.exp(-(self.k * np.pi / self.L)**2 * t.reshape(1, -1))  # (d, n_t)
#         u_xt = np.einsum("k,kt,kx->xt", coeffs, decay, phi_k_x)  # (n_x, n_t)
#         return u_xt
#
#     def generate_data(self, initial: Union[Callable[[np.ndarray], np.ndarray], np.ndarray, None] = None, 
#                       noise: float = 0.5, generate_from_coefficients: bool = False) -> None:
#         """
#         Generates synthetic data from the specified or configured initial condition and stores it in self.data_vector and self.data.
#
#         Args:
#             initial: Callable u_0(x) or array of Fourier coefficients. If None, uses self.config.true_initial.
#             noise: Noise standard deviation. If None, uses self.config.noise_level.
#             generate_from_coefficients: If True, forces initial to be treated as coefficients (if array).
#
#         Raises:
#             ValueError: If required attributes are missing or inputs are invalid.
#         """
#         if not hasattr(self, 'config'):
#             raise ValueError("self.config must be defined")
#         if not hasattr(self, 'x_obs') or not hasattr(self, 't_obs'):
#             raise ValueError("self.x_obs and self.t_obs must be defined")
#         if initial is None and not callable(self.config.true_initial):
#             raise ValueError("self.config.true_initial must be a callable function if initial is None")
#         
#         initial = self.config.true_initial if initial is None else initial
#         noise_level = self.config.noise_level if noise is None else noise
#
#         if isinstance(initial, np.ndarray) or generate_from_coefficients:
#             if not isinstance(initial, np.ndarray):
#                 raise ValueError("initial must be a NumPy array when generate_from_coefficients is True")
#             if initial.shape != (len(self.k),):
#                 raise ValueError(f"initial array must have shape ({len(self.k)},), got {initial.shape}")
#             u_clean = self.solve_from_coefficients(initial, self.x_obs, self.t_obs)
#         else:
#             if not callable(initial):
#                 raise ValueError("initial must be a callable function")
#             u_clean = self.solve_from_initial(initial, self.x_obs, self.t_obs)
#
#         rng = np.random.default_rng(seed=self.config.seed)
#         noise = noise_level * rng.normal(size=u_clean.shape)
#         self.data = u_clean + noise
#         self.data_vector = self.data.flatten()
#
#     def log_likelihood(self, x: KLSample) -> np.float64:
#         """
#         Compute log likelihood for heat equation observations using KL sample.
#         
#         Args:
#             x: KLSample object containing KL coefficients (xi_k) and callable u_0(x).
#             alpha: Covariance exponent for C_0 = A^(-alpha).
#         
#         Returns:
#             log_lik: Log likelihood value.
#         """
#         # Compute KL scaling: sqrt(mu_k) = sqrt(lambda_k^(-alpha)) = (k * pi / L)^(-alpha)
#         lambda_k = (self.k * np.pi / self.L) ** 2
#         kl_scaling = lambda_k ** (-self.config.alpha / 2)  # (k * pi / L)^(-alpha)
#         # Adjust for unnormalized basis: coeffs = xi_k * (k * pi / L)^(-alpha) * sqrt(2/L)
#         coeffs = x.coefficients * kl_scaling * np.sqrt(2 / self.L)
#         
#         # Solve heat equation using coefficients
#         predicted = self.solve_from_coefficients(coeffs, self.x_obs, self.t_obs)
#         
#         # Compute log likelihood
#         residuals = self.data - predicted
#         sigma2 = self.variance ** 2
#         log_lik = -0.5 * np.sum(residuals ** 2) / sigma2
#         log_lik -= 0.5 * residuals.size * np.log(2 * np.pi * sigma2)
#         return np.float64(log_lik)
#
#     def log_prior(self, func) -> np.float64:
#         """
#         Evaluate prior density of the function.
#         Uses coefficients if available; otherwise, falls back to functional evaluation.
#         """
#         if hasattr(func, 'coefficients'):
#             phi = func.coefficients
#             d = len(phi)
#             return -0.5 * np.sum(phi ** 2) - 0.5 * d * np.log(2 * np.pi)
#         else:
#             values = func(self.x_obs)
#             prior = values @ self.prior_precision @ values
#             return -0.5 * prior - 0.5 * len(values) * np.log(2 * np.pi)
#
#     def _construct_prior_precision(self) -> np.ndarray:
#         """
#         Construct a diagonal approximation of the precision matrix over x_obs.
#         This is a placeholder: for true consistency, use KL basis projection.
#         """
#         n = len(self.x_obs)
#         return np.eye(n)  # Identity prior as placeholder
#
#
# # === 2. Configuration for PCN sampler ===
# @dataclass(frozen=True)
# class PCNConfig:
#     target_distribution: TargetDistribution
#     x_grid: np.ndarray
#     domain_length: float
#     alpha: float
#     beta: float
#     n_terms: int
#     burn_in: int = 200
#
#     def __post_init__(self):
#         if not 0 < self.beta < 1:
#             raise ValueError("beta must be in (0, 1)")
#         if self.alpha <= 0.5:
#             raise ValueError("alpha must be > 0.5")
#         if self.n_terms < 1:
#             raise ValueError("n_terms must be ≥ 1")
#         if self.burn_in < 0:
#             raise ValueError("burn_in must be ≥ 0")
#
# # === 3. Run the inverse problem experiment ===
#
# def run_heat_equation_inverse_problem():
#     """Function to run the heat equation inverse problem with PCN sampling."""
#     
#     # Spatial domain
#     L = 1.0
#     x_grid = np.linspace(0, L, 128)
#     x_obs = np.linspace(0, L, 30)
#     t_obs = np.array([0.1])  # Make t_obs a numpy array
#     
#     # Create configuration
#     config = HeatEquationTargetConfig(
#         x_grid=x_grid,
#         x_obs=x_obs,
#         t_obs=t_obs,
#         true_initial=lambda x: np.sin(2 * np.pi * x),
#         noise_level=0.07,
#         variance=0.05,
#         k=400,
#         seed=42,
#         alpha=2
#     )
#     
#     # Create target
#     target = config.create()
#     
#     # PCN configuration
#     pcn_config = PCNConfig(
#         target_distribution=target,
#         x_grid=x_grid,
#         domain_length=L,
#         alpha=target.alpha,
#         beta=0.2,
#         n_terms=400,
#         burn_in=200
#     )
#     
#     # Run sampler
#     sampler = PCN(pcn_config)
#     sampler(1000)
#     sampler.burn(pcn_config.burn_in)
#     
#     # Posterior mean estimate
#     posterior_mean = np.mean([s.function(x_grid) for s in sampler.chain], axis=0)
#     true_u0 = target.true_initial(x_grid)
#     
#     # Plot results
#     plt.figure(figsize=(10, 6))
#     plt.plot(x_grid, true_u0, '--', label='True initial condition')
#     plt.plot(x_grid, posterior_mean, label='Posterior mean estimate')
#     plt.title("Bayesian Inverse Heat Problem via PCN")
#     plt.xlabel("$x$")
#     plt.ylabel("$u_0(x)$")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     
#     return target, sampler, posterior_mean
#
# if __name__ == "__main__":
#     run_heat_equation_inverse_problem()
#
from typing import Callable, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from Library.LibMCMC.pCN2 import KarhunenLoeveExpansion, PCN, KLSample, PCNProposal, PCNConfig
from Library.LibMCMC.Distributions import TargetDistribution
from Library.LibMCMC.PRNG import RNG, SEED

# === 1. Heat Equation Target Classes ===

from dataclasses import dataclass, field
from typing import Callable, Union, Tuple
import numpy as np
import time
from contextlib import contextmanager
from typing import Dict

# Global dictionary to store timing data
timing_data: Dict[str, float] = {}

@contextmanager
def time_block(label: str):
    """Context manager to time a code block and store/print results."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timing_data[label] = timing_data.get(label, 0.0) + elapsed
        print(f"Operation: {label}, Time: {elapsed:.4f} seconds")

@dataclass
class HeatEquationTargetConfig:
    x_grid: np.ndarray
    x_obs: np.ndarray
    t_obs: np.ndarray
    true_initial: Callable[[np.ndarray], np.ndarray] = lambda x: np.sin(2 * np.pi * x)
    noise_level: float = 0.07
    variance: float = 0.05
    kth_mode: int = 400
    seed: int = 42
    alpha: float = 4

    fourier_coefficients: np.ndarray = field(default=None, init=False)
    data_vector: np.ndarray = field(default=None, init=False)
    L: float = field(default=None, init=False)
    k: np.ndarray = field(default=None, init=False)


    lambda_k: np.ndarray = field(default=None, init=False)
    kl_scaling: np.ndarray = field(default=None, init=False)
    sqrt2_over_L: float = field(default=None, init=False)
    phi_k_x_obs: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        self.L = self.x_grid[-1] - self.x_grid[0]
        self.k = np.arange(1, self.kth_mode + 1).reshape(-1, 1)

        self.lambda_k = (self.k * np.pi / self.L) ** 2  # shape (kth_mode, 1)
        self.kl_scaling = self.lambda_k.squeeze() ** (-self.alpha / 2)  # shape (kth_mode,)
        self.sqrt2_over_L = np.sqrt(2 / self.L)
        self.phi_k_x_obs = self.sqrt2_over_L * np.sin(self.k * np.pi * self.x_obs.reshape(1, -1) / self.L)  # (k, n_obs)

    def create(self) -> "HeatEquationTarget":
        target = HeatEquationTarget(self)
        target.generate_data()
        return target

    def get_size(self) -> Tuple[int, int]:
        return (len(self.x_obs), len(self.t_obs))

class HeatEquationTarget(TargetDistribution):
    def __init__(self, config: HeatEquationTargetConfig):
        self.config = config
        self.x_obs = config.x_obs
        self.t_obs = config.t_obs
        self.variance = config.variance
        self.L = config.L
        self.k = config.k
        self.true_initial = config.true_initial
        self.alpha = config.alpha
        self.noise_level = config.noise_level
        self.kl_scaling = config.kl_scaling
        self.sqrt2_over_L = config.sqrt2_over_L
        self.phi_k_x_obs = config.phi_k_x_obs


        self.generate_data(generate_from_coefficients=False)

        super().__init__(
            prior=None,
            likelihood=None,
            data=self.data_vector,
            sigma=self.noise_level
        )
        self.prior_precision = np.eye(len(self.x_obs))  # Placeholder

    def solve_from_initial(self, u0: Callable[[np.ndarray], np.ndarray], x: np.ndarray, t: np.ndarray) -> np.ndarray:
        with time_block("HeatEquationTargetConfig.1"):
            u0_vals = u0(self.x_obs).reshape(1, -1)
            dx = self.x_obs[1] - self.x_obs[0]
            coeffs = np.sum(u0_vals * self.phi_k_x_obs, axis=1) * dx
            return self.solve_from_coefficients(coeffs, x, t)

    def solve_from_coefficients(self, coeffs: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        with time_block("HeatEquationTargetConfig.2"):
            assert coeffs.ndim == 1 and x.ndim == 1 and t.ndim == 1
            d = len(coeffs)  # number of active KL/Fourier modes
            phi_k_x = np.sin(self.k * np.pi * x.reshape(1, -1) / self.L)
            decay = np.exp(-(self.k * np.pi / self.L) ** 2 * t.reshape(1, -1))
            return np.einsum("k,kt,kx->xt", coeffs, decay, phi_k_x)

    def solve_heat_equation_from_initial(self, u0: Callable[[np.ndarray], np.ndarray], x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Solve the heat equation with initial condition u₀(x) using Fourier spectral method.
        Args:
            u0: Callable function u₀(x) defining the initial condition.
            x: 1D array of spatial evaluation points.
            t: 1D array of time evaluation points.
        Returns:
            u_xt: Array of shape (len(x), len(t)) containing u(x, t).
        """
        assert x.ndim == 1 and t.ndim == 1, "x and t must be 1D arrays"
        
        # Handle observation points
        x_obs = self.x_obs
        
        # Define spectral basis (same as Laplacian eigenfunctions)
        k = self.k  # shape (d,)
        
        # Compute basis functions at observation points for projection
        phi_k_obs = np.sin(k.reshape(-1, 1) * np.pi * x_obs.reshape(1, -1) / self.L)  # (d, n_obs)
        
        # Evaluate initial condition at observation points
        u0_vals = u0(x_obs)  # (n_obs,)
        
        # Project initial condition onto basis using scipy.integrate.trapezoid
        # (replacing deprecated np.trapz)
        from scipy.integrate import trapezoid
        coeffs = trapezoid(u0_vals * phi_k_obs, x_obs, axis=1) * (2/self.L)  # shape (d,)
        
        # Create basis functions at evaluation points
        phi_k_x = np.sin(k.reshape(-1, 1) * np.pi * x.reshape(1, -1) / self.L)  # (d, n_x)
        
        # Time evolution: exponential decay of each mode
        lambda_k = (k * np.pi / self.L) ** 2  # Eigenvalues of Laplacian
        decay = np.exp(-lambda_k.reshape(-1, 1) * t.reshape(1, -1))  # (d, n_t)
        
        # Reconstruct solution at all space-time points - ensure t is first dimension for output
        u_xt = np.zeros((len(x), len(t)))
        for i in range(len(k)):
            u_xt += coeffs[i] * np.outer(phi_k_x[i], decay[i])
        
        return u_xt

    def generate_data(self, initial=None, noise_level: float = None, generate_from_coefficients: bool = False) -> None:
        initial = self.config.true_initial if initial is None else initial
        noise_level = self.config.noise_level if noise_level is None else noise_level
        rng = np.random.default_rng(seed=self.config.seed)

        if generate_from_coefficients:
            if not isinstance(initial, np.ndarray):
                raise ValueError("Initial must be a NumPy array when generate_from_coefficients is True")
            u_clean = self.solve_from_coefficients(initial, self.x_obs, self.t_obs)  # should return shape (n_obs,)
            assert u_clean.shape == self.x_obs.shape
        elif callable(initial):
            u_clean = self.solve_heat_equation_from_initial(initial, self.x_obs, self.t_obs)  # shape (n_obs,)
        else:
            raise TypeError("Initial must be a callable or coefficient vector")

        noise = noise_level * rng.normal(size=u_clean.shape)  # one noise per scalar obs
        self.data = u_clean + noise
        self.data_vector = self.data.flatten()

    def log_likelihood(self, x) -> np.float64:
        # with time_block("HeatEquationTargetConfig.5"):
        # if hasattr(x, 'coefficients'):
        #     coeffs = x.coefficients.squeeze()
        # else:
        #     coeffs = x.squeeze()

        # Project KL coefficients into Fourier coefficients
        # scaled_coeffs = coeffs * self.kl_scaling * self.sqrt2_over_L

        # Evaluate predicted data using the forward model at observation points
        predicted = self.solve_heat_equation_from_initial(x.function, self.x_obs, self.t_obs)

        # Compute residual and Gaussian log-likelihood
        residuals = self.data - predicted
        sigma2 = self.variance ** 2
        log_lik = -0.5 * np.sum(residuals ** 2) / sigma2
        log_lik -= 0.5 * residuals.size * np.log(2 * np.pi * sigma2)
        return np.float64(log_lik)

    def log_prior(self, func) -> np.float64:
        # with time_block("HeatEquationTargetConfig.6"):
        if hasattr(func, 'coefficients'):
            phi = func.coefficients
        elif isinstance(func, np.ndarray):
            phi = func
        elif callable(func):
            values = func(self.x_obs)
            prior = values @ self.prior_precision @ values
            return -0.5 * prior - 0.5 * len(values) * np.log(2 * np.pi)
        else:
            raise TypeError(f"Unsupported type for log_prior: {type(func)}")

        d = len(phi)
        return -0.5 * np.sum(phi ** 2) - 0.5 * d * np.log(2 * np.pi)

    def _construct_prior_precision(self) -> np.ndarray:
        with time_block("HeatEquationTargetConfig.7"):
            return np.eye(len(self.x_obs))  # Identity placeholder


# # === 2. Configuration for PCN sampler ===
# # @dataclass(frozen=True)
# # class PCNConfig:
# #     target_distribution: TargetDistribution
# #     x_grid: np.ndarray
# #     domain_length: float
# #     alpha: float
# #     beta: float
# #     n_terms: int
# #     burn_in: int = 200
# #
# #     def __post_init__(self):
# #         if not 0 < self.beta < 1:
# #             raise ValueError("beta must be in (0, 1)")
# #         if self.alpha <= 0.5:
# #             raise ValueError("alpha must be > 0.5")
# #         if self.n_terms < 1:
# #             raise ValueError("n_terms must be ≥ 1")
# #         if self.burn_in < 0:
# #             raise ValueError("burn_in must be ≥ 0")
#
# # === 3. Run the inverse problem experiment ===
#
def run_heat_equation_inverse_problem():
    """Function to run the heat equation inverse problem with PCN sampling."""
    
    # Spatial domain
    L = 1.0
    x_grid = np.linspace(0, L, 500)
    x_obs = np.linspace(0, L, 1000)
    t_obs = np.linspace(0, 0.0000002, 2)
    terms = 5000

    print("Setting up heat equation inverse problem...")
    print(f"Domain: [0, {L}]")
    print(f"Grid size: {len(x_grid)} points")
    print(f"Observation points: {len(x_obs)} locations")
    
    # Create configuration
    config = HeatEquationTargetConfig(
        x_grid=x_grid,
        x_obs=x_obs,
        t_obs=t_obs,
        true_initial=lambda x: np.sin(np.pi * x),
        noise_level=0.007,
        variance=0.2,
        kth_mode=terms,
        seed=2,
        alpha=4
    )
    
    # Create target
    target = config.create()
    
    print("Generated synthetic data...")
    print(f"Target alpha: {target.alpha}")
    print(f"Data shape: {target.data.shape}")
    
    # PCN configuration
    pcn_config = PCNConfig(
        target_distribution=target,
        x_grid=x_grid,
        domain_length=L,
        alpha=target.alpha, 
        beta=0.4,
        n_terms=50,
        burn_in=20
    )
    
    print("Created PCN configuration...")
    print(f"Beta: {pcn_config.beta}")
    print(f"Number of KL terms: {pcn_config.n_terms}")
    
    # Run sampler
    print("Initializing PCN sampler...")
    sampler = PCN(pcn_config)
    
    print("Running PCN sampler for 500 iterations...")
    try:
        sampler(1000)
        print("Sampling completed successfully")
        
        print(f"Burning {pcn_config.burn_in} samples...")
        sampler.burn(pcn_config.burn_in)
    except Exception as e:
        print(f"Error during sampling: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Posterior mean estimate
    posterior_mean = np.mean([s.function(x_grid) for s in sampler.chain], axis=0)
    true_u0 = target.true_initial(x_grid)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, true_u0, '--', label='True initial condition')
    plt.plot(x_grid, posterior_mean, label='Posterior mean estimate')
    plt.title("Bayesian Inverse Heat Problem via PCN")
    plt.xlabel("$x$")
    plt.ylabel("$u_0(x)$")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Evaluate true initial condition
    true_u0 = target.true_initial(x_grid)

    # Evaluate function samples from MCMC chain on the visualization grid
    num_display = 30  # how many samples to show
    sample_functions = [s.function(x_grid) for s in sampler.chain[:num_display]]

    # Posterior mean estimate
    posterior_mean = np.mean(sample_functions, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    for u_sample in sample_functions:
        plt.plot(x_grid, u_sample, color='gray', alpha=0.3, linewidth=1)

    # Overlay posterior mean and true initial condition
    plt.plot(x_grid, true_u0, '--', color='black', label='True initial condition', linewidth=2)
    plt.plot(x_grid, posterior_mean, color='blue', label='Posterior mean', linewidth=2)

    plt.title("Posterior Function Samples (pCN)")
    plt.xlabel("$x$")
    plt.ylabel("$u_0(x)$")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return target, sampler, posterior_mean

if __name__ == "__main__":
    run_heat_equation_inverse_problem()
