from typing import Callable, Tuple, Union
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import multiprocessing as mp
from functools import partial
import time
from contextlib import contextmanager
from typing import Dict
# import cupy as cp  # Add CuPy for GPU acceleration

# Global flag to control GPU usage
USE_GPU = True

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

# GPU Utilities
def to_gpu(arr):
    """Transfer numpy array to GPU if GPU is enabled"""
    if USE_GPU:
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    """Transfer GPU array back to CPU if needed"""
    if USE_GPU and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

def xp_select():
    """Return the appropriate array library based on GPU flag"""
    return cp if USE_GPU else np

# GPU-accelerated Heat Equation Solver
def solve_heat_equation_from_initial_gpu(u0, x, t, x_obs, L, k):
    """
    GPU-accelerated implementation of the heat equation solver
    """
    xp = xp_select()
    
    # Transfer arrays to GPU if not already there
    x_gpu = to_gpu(x)
    t_gpu = to_gpu(t)
    x_obs_gpu = to_gpu(x_obs)
    k_gpu = to_gpu(k)
    
    # Evaluate initial condition at observation points
    u0_vals = to_gpu(u0(to_cpu(x_obs_gpu)))
    
    # Compute basis functions
    phi_k_obs = xp.sin(k_gpu.reshape(-1, 1) * xp.pi * x_obs_gpu.reshape(1, -1) / L)
    
    # Project initial condition onto basis
    # Since trapezoid is not available in CuPy, we'll implement it directly
    dx = x_obs_gpu[1] - x_obs_gpu[0]
    y = u0_vals * phi_k_obs
    coeffs = xp.sum(y[:, :-1] + y[:, 1:], axis=1) * dx * 0.5 * (2/L)
    
    # Create basis functions at evaluation points
    phi_k_x = xp.sin(k_gpu.reshape(-1, 1) * xp.pi * x_gpu.reshape(1, -1) / L)
    
    # Time evolution
    lambda_k = (k_gpu * xp.pi / L) ** 2
    decay = xp.exp(-lambda_k.reshape(-1, 1) * t_gpu.reshape(1, -1))
    
    # Reconstruct solution - vectorized implementation
    u_xt = xp.zeros((len(x_gpu), len(t_gpu)))
    
    # Use einsum for efficient matrix multiplication
    u_xt = xp.einsum('k,kt,kx->xt', coeffs, decay, phi_k_x)
    
    # Transfer result back to CPU if needed
    return to_cpu(u_xt)

# Replace the original heat equation solver method with a GPU-accelerated version
def gpu_accelerated_heat_equation_solver(self, u0, x, t):
    """Wrapper to call the GPU-accelerated solver"""
    return solve_heat_equation_from_initial_gpu(
        u0, x, t, 
        self.x_obs, self.L, self.k
    )


def covariance_weighted_least_squares(functions, target) -> float:
    """
    Compute the covariance-weighted least squares error:
        ‖μ_post - μ_true‖²_{Γ⁻¹}
    where μ_post is the posterior predictive mean in data space
    and μ_true is the noise-free true forward output.

    Args:
        functions: List of posterior samples u₀^{(j)} (callable functions)
        target: HeatEquationTarget, which defines the forward model, noise, and true initial condition

    Returns:
        Scalar value: covariance-weighted least squares error
    """
    # Predictive samples pushed forward
    pred_vals = np.array([
        target.solve_heat_equation_from_initial(f, target.x_obs, target.t_obs).flatten()
        for f in functions
    ])
    mu_post = np.mean(pred_vals, axis=0)

    # True (noise-free) forward solution
    mu_true = target.solve_heat_equation_from_initial(target.true_initial, target.x_obs, target.t_obs).flatten()

    # Noise covariance is diagonal: Γ = σ² I ⇒ Γ⁻¹ = (1/σ²) I
    sigma2 = target.variance ** 2
    diff = mu_post - mu_true

    return np.dot(diff, diff) / sigma2

def kl_divergence_gaussian(mu1: np.ndarray, sigma1: np.ndarray, 
                          mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Compute the KL divergence between two multivariate Gaussians.
    
    Args:
        mu1: Mean of the first Gaussian distribution
        sigma1: Covariance matrix of the first Gaussian distribution
        mu2: Mean of the second Gaussian distribution
        sigma2: Covariance matrix of the second Gaussian distribution
        
    Returns:
        KL divergence D_KL(p₁ ‖ p₂)
    """
    # Dimensionality
    m = len(mu1)
    
    # Precompute inverse of sigma2
    sigma2_inv = np.linalg.inv(sigma2)
    
    # Compute trace term
    trace_term = np.trace(sigma2_inv @ sigma1)
    
    # Compute mean difference term
    diff = mu2 - mu1
    mean_term = diff.T @ sigma2_inv @ diff
    
    # Compute log determinant term
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    sign2, logdet2 = np.linalg.slogdet(sigma2)
    logdet_term = logdet2 - logdet1
    
    # Compute KL divergence
    kl = 0.5 * (trace_term + mean_term - m + logdet_term)
    
    return kl

# Add these functions to your codebase

def compute_posterior_statistics(functions, x_grid):
    """
    Compute posterior statistics for function samples.
    
    Args:
        functions: List of callable functions (posterior samples)
        x_grid: Grid points for evaluation
        
    Returns:
        mean: Posterior mean function (callable)
        std: Standard error function (callable)
        lower: Lower credible bound function (callable)
        upper: Upper credible bound function (callable)
    """
    # Simple function wrappers as you wanted
    def mean_func(x):
        return sum(f(x) for f in functions) / len(functions)
    
    def std_func(x):
        mean_val = mean_func(x)
        return np.sqrt(sum((f(x) - mean_val)**2 for f in functions) / len(functions))
    
    def lower_func(x):
        return mean_func(x) - 1.96 * std_func(x)
    
    def upper_func(x):
        return mean_func(x) + 1.96 * std_func(x)
    
    return mean_func, std_func, lower_func, upper_func


def compute_posterior_predictive_statistics(initial_functions, target, x_grid, t_grid):
    """
    Compute posterior predictive statistics.
    
    Args:
        initial_functions: List of callable initial condition functions
        target: Heat equation target with solver
        x_grid: Spatial points
        t_grid: Time points
        
    Returns:
        predictive_mean: Function u(x,t) giving posterior predictive mean
        predictive_std: Function giving standard error
        predictive_lower: Function giving lower credible bound
        predictive_upper: Function giving upper credible bound
    """
    def predictive_mean(x, t):
        x_array = np.atleast_1d(x)
        t_array = np.atleast_1d(t)
        solutions = [
            target.solve_heat_equation_from_initial(f, x_array, t_array) 
            for f in initial_functions
        ]
        return sum(solutions) / len(initial_functions)
    
    def predictive_std(x, t):
        x_array = np.atleast_1d(x)
        t_array = np.atleast_1d(t)
        solutions = [
            target.solve_heat_equation_from_initial(f, x_array, t_array) 
            for f in initial_functions
        ]
        mean_val = sum(solutions) / len(initial_functions)
        return np.sqrt(sum((s - mean_val)**2 for s in solutions) / len(initial_functions))
    
    def predictive_lower(x, t):
        return predictive_mean(x, t) - 1.96 * predictive_std(x, t)
    
    def predictive_upper(x, t):
        return predictive_mean(x, t) + 1.96 * predictive_std(x, t)
    
    return predictive_mean, predictive_std, predictive_lower, predictive_upper

def plot_samples_with_credible_region(x_grid, functions, true_function=None, 
                                     n_samples=10, confidence=0.95,
                                     title="Function Samples and Credible Region", 
                                     figsize=(10, 6)):
    """Basic plot for functions with credible regions"""
    import matplotlib.pyplot as plt
    
    # Compute mean and credible interval
    mean_func, std_func, lower_func, upper_func = compute_posterior_statistics(functions, x_grid)
    
    # Evaluate on grid for plotting
    mean_vals = mean_func(x_grid)
    lower_vals = lower_func(x_grid)
    upper_vals = upper_func(x_grid)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot credible region
    ax.fill_between(x_grid, lower_vals, upper_vals, alpha=0.3, color='blue',
                   label=f'{int(confidence*100)}% Credible Region')
    
    # Plot samples
    if n_samples > 0:
        indices = np.random.choice(len(functions), size=min(n_samples, len(functions)), replace=False)
        for idx in indices:
            sample_vals = np.array([functions[idx](x) for x in x_grid])
            ax.plot(x_grid, sample_vals, 'gray', alpha=0.3, linewidth=0.5)
    
    # Plot mean
    ax.plot(x_grid, mean_vals, 'b-', linewidth=2, label='Mean')
    
    # Plot true function if provided
    if true_function is not None:
        true_vals = np.array([true_function(x) for x in x_grid])
        ax.plot(x_grid, true_vals, 'r--', linewidth=2, label='True Function')
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_predictive_vs_data(x_grid: np.ndarray, t_fixed: float, 
                            pred_mean, pred_lower, pred_upper,
                            synthetic_data: np.ndarray, 
                            title: str = "Posterior Predictive vs Data"):
    """
    Plot posterior predictive mean, credible bands, and data at a fixed time point.
    
    Args:
        x_grid: Spatial points
        t_fixed: Fixed time value for evaluation
        pred_mean: Function for posterior predictive mean
        pred_lower: Function for lower credible bound
        pred_upper: Function for upper credible bound
        synthetic_data: Array of observed/synthetic data at the fixed time
        title: Plot title
        
    Returns:
        fig: The matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Evaluate statistical functions at the fixed time
    t_array = np.array([t_fixed])
    mean_vals = pred_mean(x_grid, t_array).squeeze()
    lower_vals = pred_lower(x_grid, t_array).squeeze()
    upper_vals = pred_upper(x_grid, t_array).squeeze()
    
    # Plot data
    ax.scatter(x_grid, synthetic_data, color='red', s=15, label="Observed Data", zorder=3)
    
    # Plot predictive mean
    ax.plot(x_grid, mean_vals, color='blue', linewidth=2, label="Predictive Mean", zorder=2)
    
    # Plot credible region
    ax.fill_between(x_grid, lower_vals, upper_vals, color='blue', alpha=0.3, 
                   label="95% Credible Band", zorder=1)
    
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend()
    ax.grid(True)
    
    return fig


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


    def generate_predictive_function(self, u0: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return a callable (x, t) ↦ u(x, t) for given initial condition."""
        def u_xt(x: np.ndarray, t: np.ndarray) -> np.ndarray:
            return self.solve_heat_equation_from_initial(u0, x, t)
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


########################################################################################################################## Modelling side



def run_heat_equation_inverse_problem():
    """Function to run the heat equation inverse problem with PCN sampling."""
    
    # Spatial domain
    L = 1.0
    x_grid = np.linspace(0, L, 500)
    x_obs = np.linspace(0, L, 1000)
    t_obs = np.linspace(0, 0.0002, 100)
    terms = 10

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
        noise_level=0.10,
        variance=1,
        kth_mode=terms,
        seed=2,
        alpha=3
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
        beta=0.3,
        n_terms=10,
        burn_in=50
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

    # Posterior mean estimate

    # Plo    
    return target, sampler, posterior_mean

def analyze_heat_equation_inference(target, sampler, x_grid=None, confidence=0.95):
    """
    Analyze the results of heat equation inverse problem inference.
    
    Args:
        target: HeatEquationTarget instance
        sampler: PCN sampler instance with posterior samples
        x_grid: Grid for evaluation (default: None, uses sampler.x_grid)
        confidence: Confidence level for credible intervals
        
    Returns:
        Dictionary of analysis results
    """
    if x_grid is None:
        x_grid = sampler.x_grid
    
    # Get posterior samples as functions
    posterior_functions = [s.function for s in sampler.chain]
    
    # 1. Create statistical function wrappers for initial condition
    mean_function = mean_function_wrapper(posterior_functions)
    std_function = std_function_wrapper(posterior_functions)
    lower_function, upper_function = credible_band_function_wrappers(posterior_functions, confidence)
    
    # 2. Generate posterior predictive functions
    predictive_functions = []
    for func in posterior_functions:
        def pred_func(x, t, ic=func):
            # Make sure x and t are numpy arrays
            x_array = np.atleast_1d(x)
            t_array = np.atleast_1d(t)
            return target.solve_heat_equation_from_initial(ic, x_array, t_array)
        predictive_functions.append(pred_func)
    
    # 3. Create statistical function wrappers for posterior predictive
    pred_mean_func = mean_function_wrapper(predictive_functions)
    pred_lower_func, pred_upper_func = credible_band_function_wrappers(predictive_functions, confidence)
    
    # 4. Plot initial condition inference
    fig_init = plot_initial_condition_credible_region(
        sampler, 
        x_grid=x_grid, 
        true_initial=target.true_initial,
        confidence=confidence
    )
    
    # 5. Plot posterior predictive vs data
    # Create finer evaluation grid for plotting
    x_eval = np.linspace(0, target.L, 200)
    t_eval = np.linspace(0, np.max(target.t_obs), 4)  # 4 time points for visualization
    
    fig_pred = plot_posterior_predictive_credible_region(
        sampler,
        x_eval=x_eval, 
        t_eval=t_eval,
        x_obs=target.x_obs, 
        data=target.data,
        confidence=confidence
    )
    
    # 6. Quantitative analysis if desired
    # Evaluate mean and covariance at observation points for metrics
    pred_mean = pred_mean_func(target.x_obs, target.t_obs)
    true_mean = target.solve_heat_equation_from_initial(target.true_initial, target.x_obs, target.t_obs)
    
    # Compute observation precision matrix (inverse of covariance)
    gamma_inv = np.eye(np.prod(pred_mean.shape)) / (target.variance ** 2)
    
    # Calculate error metrics
    l2_error = covariance_weighted_least_squares(
        pred_mean.flatten(), 
        true_mean.flatten(), 
        gamma_inv
    )
    
    return {
        'initial_condition_plot': fig_init,
        'posterior_predictive_plot': fig_pred,
        'l2_error': l2_error,
        'mean_function': mean_function,
        'predictive_mean_function': pred_mean_func,
        'predictive_credible_bands': (pred_lower_func, pred_upper_func)
    }

# def analyze_mcmc_results(target, sampler):
#     """Analyze MCMC results using ALL posterior samples"""
#     # Use ALL posterior samples
#     functions = [s.function for s in sampler.chain]
#     
#     # Generate plots
#     x_grid = sampler.x_grid
#     ic_plot = plot_samples_with_credible_region(
#         x_grid, 
#         functions, 
#         true_function=target.true_initial,
#         title="Initial Condition: Posterior Samples and Credible Region"
#     )
#     
#     # Calculate L2 error using ALL samples
#     mean_func, _, _, _ = compute_posterior_statistics(functions, x_grid)
#     mean_vals = np.array([mean_func(x) for x in x_grid])
#     true_vals = target.true_initial(x_grid)
#     l2_error = np.mean((mean_vals - true_vals)**2) / target.variance**2
#     
#     return {
#         'ic_plot': ic_plot,
#         'l2_error': l2_error,
#         'mean_function': mean_func
#     }

def plot_predictive_vs_data(x_grid: np.ndarray, t_fixed: float, 
                            pred_mean, pred_lower, pred_upper,
                            synthetic_data: np.ndarray, 
                            title: str = "Posterior Predictive vs Data"):
    """
    Plot posterior predictive mean, credible bands, and data at a fixed time point.
    
    Args:
        x_grid: Spatial points
        t_fixed: Fixed time value for evaluation
        pred_mean: Function for posterior predictive mean
        pred_lower: Function for lower credible bound
        pred_upper: Function for upper credible bound
        synthetic_data: Array of observed/synthetic data at the fixed time
        title: Plot title
        
    Returns:
        fig: The matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Evaluate statistical functions at the fixed time
    t_array = np.array([t_fixed])
    mean_vals = pred_mean(x_grid, t_array).squeeze()
    lower_vals = pred_lower(x_grid, t_array).squeeze()
    upper_vals = pred_upper(x_grid, t_array).squeeze()
    
    # Plot data
    ax.scatter(x_grid, synthetic_data, color='red', s=15, label="Observed Data", zorder=3)
    
    # Plot predictive mean
    ax.plot(x_grid, mean_vals, color='blue', linewidth=2, label="Predictive Mean", zorder=2)
    
    # Plot credible region
    ax.fill_between(x_grid, lower_vals, upper_vals, color='blue', alpha=0.3, 
                   label="95% Credible Band", zorder=1)
    
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend()
    ax.grid(True)
    
    return fig


def analyze_mcmc_results(target, sampler, t_fixed: float = None):
    """
    Analyze posterior samples with synthetic data at a fixed time.
    
    Args:
        target: HeatEquationTarget instance
        sampler: PCN sampler instance with posterior samples
        t_fixed: Fixed time point for analysis (default: max observation time)
        
    Returns:
        Dictionary of analysis results
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set fixed time if not provided
    if t_fixed is None:
        t_fixed = np.max(target.t_obs)
    
    # Create t array with single value for evaluation
    t_array = np.array([t_fixed])
    
    # Spatial grid for visualization
    x_grid = sampler.x_grid
    
    # Get ALL posterior samples as functions
    functions = [s.function for s in sampler.chain]
    
    # Initial condition credible region
    ic_plot = plot_samples_with_credible_region(
        x_grid,
        functions,
        true_function=target.true_initial,
        title="Initial Condition: Posterior Samples and Credible Region"
    )
    
    # Compute L² error for initial condition
    mean_func, _, _, _ = compute_posterior_statistics(functions, x_grid)
    mean_vals = mean_func(x_grid)
    true_vals = target.true_initial(x_grid)
    l2_error = np.mean((mean_vals - true_vals)**2) / target.variance**2
    
    # Generate true solution at fixed time with noise
    true_solution = target.solve_heat_equation_from_initial(target.true_initial, x_grid, t_array).squeeze()
    
    # Add noise (using target's noise level)
    rng = np.random.default_rng(seed=target.config.seed + 1)  # Different seed for new data
    synthetic_data = true_solution + target.noise_level * rng.normal(size=true_solution.shape)
    
    # Compute posterior predictive statistics functions
    predictive_mean, predictive_std, predictive_lower, predictive_upper = compute_posterior_predictive_statistics(
        functions, target, x_grid, t_array
    )
    
    # Plot predictive vs data USING THE STATISTICAL FUNCTIONS
    pred_plot = plot_predictive_vs_data(
        x_grid=x_grid,
        t_fixed=t_fixed,
        pred_mean=predictive_mean,
        pred_lower=predictive_lower,
        pred_upper=predictive_upper,
        synthetic_data=synthetic_data,
        title=f"Posterior Predictive vs Synthetic Data at t = {t_fixed:.6f}"
    )
    
    return {
        'ic_plot': ic_plot,
        'predictive_plot': pred_plot,
        'l2_error': l2_error,
        'synthetic_data': synthetic_data,
        'mean_function': mean_func,
        'predictive_mean': predictive_mean,
        'predictive_lower': predictive_lower,
        'predictive_upper': predictive_upper
    }

# Option 3: Sparse coefficients (only specific modes active)
def sparse_coeffs(num_terms, active_modes=None, values=None):
    """Generate coefficients with only specific modes active"""
    if active_modes is None:
        active_modes = [1, 3, 5]  # Default to odd harmonics
    if values is None:
        values = [1.0] * len(active_modes)  # Default to equal values
        
    coeffs = np.zeros(num_terms)
    for mode, value in zip(active_modes, values):
        if 1 <= mode <= num_terms:
            coeffs[mode-1] = value
    return coeffs
def alternating_block_coeffs(num_terms=50, block_size=10, active_blocks=None, values=None):
    """
    Generate coefficients with blocks of active and inactive modes.
    
    Args:
        num_terms: Total number of terms
        block_size: Size of each block (active or inactive)
        active_blocks: Indices of which blocks should be active (0-indexed)
        values: Values for each active block
        
    Returns:
        Array of coefficients with specified pattern
    """
    # Default: make the first block active
    if active_blocks is None:
        active_blocks = [0]
    
    # Default: all active blocks have coefficient 1.0
    if values is None:
        values = [1.0] * len(active_blocks)
    
    # Initialize all coefficients to zero
    coeffs = np.zeros(num_terms)
    
    # Set active blocks
    for block_idx, value in zip(active_blocks, values):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, num_terms)
        coeffs[start_idx:end_idx] = value
    
    return coeffs

# 10 active, 10 inactive, repeating across 50 terms
FOURIER_COEFFS = alternating_block_coeffs(
    num_terms=20, 
    block_size=5, 
    active_blocks=[0, 1, 2, 4, 5], 
    values=[1.0, 0.7, 0.4, 0.5, 0.8]
)

def true_initial_function(x):
    """
    Fourier sine series with proper orthonormal basis (including √2 factor)
    
    Args:
        x: Array of x values
        
    Returns:
        Array of function values at points x
    """
    # Get number of terms from global coefficient array
    num_terms = len(FOURIER_COEFFS)
    
    # Prepare arrays for fast computation
    n_values = np.arange(1, num_terms+1).reshape(-1, 1)
    x = np.atleast_1d(x)
    
    # Calculate all sine terms at once with √2 factor for orthonormality
    sine_matrix = np.sqrt(2) * np.sin(n_values * np.pi * x)
    
    # Multiply by coefficients and sum
    return np.dot(FOURIER_COEFFS, sine_matrix)

def run_single_chain(chain_data):
    """
    Run a single MCMC chain with the given percentage of data.
    This function will be executed in parallel.
    
    Args:
        chain_data: Dictionary with chain parameters
        
    Returns:
        Dictionary with chain results
    """
    percentage = chain_data['percentage']
    L = chain_data['L']
    x_grid = chain_data['x_grid']
    full_x_obs = chain_data['full_x_obs']
    t_obs = chain_data['t_obs']
    terms = chain_data['terms']
    base_iterations = chain_data['base_iterations']
    chain_index = chain_data['chain_index']
    
    # Calculate observation grid based on percentage
    n_obs = int(len(full_x_obs) * percentage / 100)
    x_obs = np.linspace(0, L, n_obs)
    
    print(f"Running chain {chain_index+1} with {percentage}% of data ({n_obs} observation points)")
    
    # Create configuration
    config = HeatEquationTargetConfig(
        x_grid=x_grid,
        x_obs=x_obs,
        t_obs=t_obs,
        true_initial=true_initial_function,  # Use the named function instead of lambda
        noise_level=0.10,
        variance=0.5,
        kth_mode=terms,
        seed=int(2 + chain_index),  # Different seed for each chain
        alpha=1.5
    )
   
    # Create target and run sampler
    try:
        target = config.create()
        
        pcn_config = PCNConfig(
            target_distribution=target,
            x_grid=x_grid,
            domain_length=L,
            alpha=target.alpha, 
            beta=0.4,
            n_terms=20,
            burn_in=100
        )
        
        sampler = PCN(pcn_config)
        sampler(base_iterations)
        sampler.burn(pcn_config.burn_in)
        
        # Calculate posterior mean
        functions = [s.function for s in sampler.chain]
        # We can't return the functions directly as they're not pickleable
        # Instead, evaluate them on the grid and return the mean values
        mean_vals = np.mean([f(x_grid) for f in functions], axis=0)

        weighted_error = covariance_weighted_least_squares(functions, target)
        
        return {
            'mean_vals': mean_vals,
            'percentage': percentage,
            'weighted_error': weighted_error,
            'status': 'success'
        }
    except Exception as e:
        import traceback
        print(f"Error in chain {chain_index+1} with {percentage}% data: {e}")
        traceback.print_exc()
        return {
            'percentage': percentage,
            'status': 'failed',
            'error': str(e)
        }

def run_multiple_chains_parallel(n_chains=5, base_iterations=2000, max_workers=None):
    """
    Run multiple heat equation inverse MCMC chains in parallel with logarithmically increasing data amounts.
    
    Args:
        n_chains: Number of chains to run
        base_iterations: Number of MCMC iterations for each chain
        max_workers: Maximum number of parallel processes (default: None, uses CPU count)
        
    Returns:
        Dictionary containing results from all chains
    """
    # Base spatial domain parameters
    L = 1.0
    x_grid = np.linspace(0, L, 200)
    full_x_obs = np.linspace(0, L, 1000)
    t_obs = np.linspace(0, 0.0001, 200)
    terms = 20
    
    # Calculate logarithmically increasing percentages
    # Use logspace to create values between 0.01 (1%) and 1.0 (100%)
    # Add a small offset to avoid 0% data
    min_percentage = 2.0  # Minimum percentage (1%)
    max_percentage = 100.0
    log_values = np.linspace(min_percentage, max_percentage, n_chains)
    percentages = [int(round(p)) for p in log_values]    

    # Ensure uniqueness of percentages
    percentages = sorted(list(set(percentages)))
    # If we lost some due to duplicates after rounding, add more at high end
    while len(percentages) < n_chains:
        percentages.append(100)
    
    print(f"Data percentages: {percentages}")
    
    # Prepare data for each chain
    chain_data_list = []
    for i, percentage in enumerate(percentages):
        # Ensure at least a few observation points even for lowest percentage
        n_obs = max(5, int(len(full_x_obs) * percentage / 100))
        
        chain_data = {
            'percentage': percentage,
            'L': L,
            'x_grid': x_grid,
            'full_x_obs': full_x_obs,
            't_obs': t_obs,
            'terms': terms,
            'base_iterations': base_iterations,
            'chain_index': i
        }
        chain_data_list.append(chain_data)
    
    # Determine number of processes
    if max_workers is None:
        max_workers = min(mp.cpu_count(), n_chains)
    
    print(f"Running {n_chains} chains in parallel using {max_workers} processes...")
    
    # Run chains in parallel with interrupt handling
    try:
        with mp.Pool(processes=max_workers) as pool:
            # Use map_async with a timeout to allow keyboard interrupts
            result_async = pool.map_async(run_single_chain, chain_data_list)
            
            # Wait for results with a timeout to allow keyboard interrupts
            try:
                results = result_async.get(timeout=999999)  # Long timeout
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Terminating processes...")
                pool.terminate()
                pool.join()
                print("All processes terminated.")
                return None
                
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        return None
    
    # Process results
    posterior_means = []
    data_percentages = []
    weighted_errors = []  # Store the weighted errors
    
    for result in results:
        if result['status'] == 'success':
            data_percentages.append(result['percentage'])
            posterior_means.append(result['mean_vals'])
            weighted_errors.append(result['weighted_error'])  # Collect the weighted errors
    
    # Sort results by percentage
    sorted_indices = np.argsort(data_percentages)
    data_percentages = [data_percentages[i] for i in sorted_indices]
    posterior_means = [posterior_means[i] for i in sorted_indices]
    weighted_errors = [weighted_errors[i] for i in sorted_indices]  # Sort these too
    
    # Plot all posterior means against true initial condition
    true_values = true_initial_function(x_grid)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_grid, true_values, 'k--', linewidth=2, label='True initial condition')
    
    # Plot posterior means from all chains
    colors = plt.cm.viridis(np.linspace(0, 1, len(posterior_means)))
    for i, (mean, percentage) in enumerate(zip(posterior_means, data_percentages)):
        plt.plot(x_grid, mean, color=colors[i], linewidth=1.5, 
                label=f'{percentage}% data')
    
    plt.title("Posterior Means with Logarithmically Increasing Data")
    plt.xlabel("$x$")
    plt.ylabel("$u_0(x)$")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {
        'posterior_means': posterior_means,
        'data_percentages': data_percentages,
        'x_grid': x_grid,
        'weighted_errors': weighted_errors
    }


def visualize_chain_results(results_dict):
    """
    Create comprehensive visualizations of multi-chain results with improved formatting.
    
    Args:
        results_dict: Dictionary returned by run_multiple_chains_parallel
    
    Returns:
        Dictionary of matplotlib figures
    """
    posterior_means = results_dict['posterior_means']
    data_percentages = results_dict['data_percentages']
    weighted_errors = results_dict['weighted_errors']
    x_grid = results_dict['x_grid']
    
    # Define true initial condition with proper vectorization
    true_values = true_initial_function(x_grid)
    
    # Create figure for visualizations
    fig1 = plt.figure(figsize=(16, 12))
    fig1.suptitle("Effect of Data Quantity on Posterior Inference", fontsize=16)
    
    # 1. Posterior Means Comparison (unchanged)
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.plot(x_grid, true_values, 'k--', linewidth=2, label='True initial condition')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(posterior_means)))
    for i, (mean, percentage) in enumerate(zip(posterior_means, data_percentages)):
        ax1.plot(x_grid, mean, color=colors[i], linewidth=1.5, 
                label=f'{percentage}% data')
    
    ax1.set_title("Posterior Means with Increasing Data")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$u_0(x)$")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax1.legend()
    ax1.grid(True)
    
    # 2. L² Error vs Data Percentage (LINEAR scale - not log-log)
    ax2 = fig1.add_subplot(2, 2, 2)
    
    # Check if weighted errors need to be inverted (if they're increasing with more data)
    # This is a common issue if the covariance_weighted_least_squares function 
    # is calculating higher values for better fits
    if weighted_errors[0] < weighted_errors[-1]:
        # If errors are increasing with data percentage, they might be inverted
        print("Warning: weighted errors are increasing with more data - check calculation")
    
    # Plot with LINEAR scales
    ax2.plot(data_percentages, weighted_errors, '-o', color='blue', linewidth=2, markersize=8)
    ax2.set_title("L² Error vs Data Percentage")
    ax2.set_xlabel("Data Percentage")
    ax2.set_ylabel("Covariance-Weighted Mean Squared Error")
    
    # Set evenly spaced ticks (at least 5) on both axes
    from matplotlib.ticker import MaxNLocator
    ax2.xaxis.set_major_locator(MaxNLocator(5))  # At least 5 ticks on x-axis
    ax2.yaxis.set_major_locator(MaxNLocator(5))  # At least 5 ticks on y-axis
    
    ax2.grid(True)
    
    # Rest of the code remains the same...



if __name__ == "__main__":

    # results = run_multiple_chains_parallel()

    # Or specify number of chains and processes
    results = run_multiple_chains_parallel(n_chains=8, max_workers=4)


    figures = visualize_chain_results(results)
    plt.figure(figures['main_figure'].number)
    plt.show()

    # target, sampler, posterior_mean = run_heat_equation_inverse_problem()
    # results = analyze_mcmc_results(target, sampler)
    #
    # # Show both plots
    # results['ic_plot'].tight_layout()
    # results['ic_plot'].show()
    # plt.show()
    #
    # results['posterior_predictive_plot'].tight_layout()
    # results['posterior_predictive_plot'].show()
    # plt.show()
    #
    # # Print L² error
    # print(f"L2 error: {results['l2_error']:.6f}")
