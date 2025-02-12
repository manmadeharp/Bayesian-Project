import numpy as np
import scipy as sp
from typing import List, Optional
from Library.Distributions import TargetDistribution, Proposal
from Library.MALA import MALA
from Library.Diagnostics import MCMCDiagnostics
from Library.MetropolisHastings import AdaptiveMetropolisHastings, MetropolisHastings
import matplotlib.pyplot as plt

class HeatDiffusionInverse:
    """
    Implementation of the Heat Diffusion inverse problem using MCMC.
    """
    def __init__(
        self, 
        x_obs: np.ndarray,  # Spatial observation points
        t_obs: float,       # Time at which observations were made
        y_obs: np.ndarray,  # Temperature observations
        N: int = 10,        # Number of Fourier coefficients
        D: float = 1.0,     # Diffusion coefficient
        l: float = 1.0,     # Length of spatial domain
        sigma: float = 0.1  # Observation noise standard deviation
    ):
        self.x_obs = x_obs
        self.t_obs = t_obs
        self.y_obs = y_obs
        self.N = N
        self.D = D
        self.l = l
        self.sigma = sigma
        
        # Compute the observation matrix Omega
        self.Omega = self._compute_omega_matrix()
        
    def _compute_omega_matrix(self) -> np.ndarray:
        """
        Compute the Omega matrix where Omega[i,j] = sin(ω_j x_i)exp(-D(jπ/l)²t)
        """
        M = len(self.x_obs)  # Number of spatial observations
        Omega = np.zeros((M, self.N))
        
        for i in range(M):
            for j in range(self.N):
                n = j + 1  # Since we start from n=1
                omega_n = n * np.pi / self.l
                Omega[i,j] = np.sin(omega_n * self.x_obs[i]) * \
                            np.exp(-self.D * (omega_n**2) * self.t_obs)
        
        return Omega

    def fundamental_solution(self, x: np.ndarray, t: float, g: callable) -> np.ndarray:
        """
        Compute the fundamental solution using convolution
        """
        dy = 0.01  # Integration step size
        y = np.arange(-5, 5, dy)  # Integration domain
        u = np.zeros_like(x)
        
        for i, xi in enumerate(x):
            # Compute the heat kernel
            kernel = 1/np.sqrt(4*np.pi*t) * np.exp(-(xi - y)**2 / (4*t))
            # Compute the convolution with initial condition g
            u[i] = np.sum(kernel * g(y)) * dy
            
        return u

    def fourier_solution(self, x: np.ndarray, t: float, A: np.ndarray) -> np.ndarray:
        """
        Compute the Fourier series solution with given coefficients
        """
        u = np.zeros_like(x)
        for n in range(1, len(A) + 1):
            omega_n = n * np.pi / self.l
            u += A[n-1] * np.sin(omega_n * x) * \
                 np.exp(-self.D * (omega_n**2) * t)
        return u

class HeatDiffusionTarget(TargetDistribution):
    """
    Target distribution for the heat diffusion inverse problem
    """
    def __init__(
        self,
        heat_model: HeatDiffusionInverse,
        prior_std: float = 1.0
    ):
        # Initialize with standard normal prior and likelihood
        super().__init__(
            prior=sp.stats.norm(loc=0, scale=prior_std),
            likelihood=sp.stats.norm,
            data=heat_model.y_obs,
            sigma=heat_model.sigma
        )
        self.heat_model = heat_model
        
    def log_likelihood(self, A: np.ndarray) -> np.float64:
        """
        Compute log likelihood: log p(y|A) where y are observations
        """
        # Compute model prediction: Ω·A
        pred = self.heat_model.Omega @ A
        
        # Compute log likelihood using Gaussian noise model
        log_lik = -0.5 * len(self.data) * np.log(2*np.pi*self.data_sigma**2)
        log_lik -= 0.5 * np.sum((self.data - pred)**2) / self.data_sigma**2
        
        return np.float64(log_lik)
        
    def log_prior(self, A: np.ndarray) -> np.float64:
        """
        Compute log prior: sum of log p(A_n) for each coefficient
        """
        # Independent normal priors for each coefficient
        return np.sum(self.prior.logpdf(A))

def run_heat_diffusion_mcmc(
    x_obs: np.ndarray,
    t_obs: float,
    y_obs: np.ndarray,
    n_samples: int = 100000,
    n_fourier: int = 10,
    step_size: float = 0.05
) -> tuple:
    """
    Run MCMC sampling for the heat diffusion inverse problem
    """
    # Initialize the model
    heat_model = HeatDiffusionInverse(
        x_obs=x_obs,
        t_obs=t_obs,
        y_obs=y_obs,
        N=n_fourier
    )
    
    # Create target distribution
    target = HeatDiffusionTarget(heat_model)
    
    prop = Proposal(
        sp.stats.multivariate_normal,
        np.eye(4)
    )

    # Initialize state with zeros
    initial_state = np.zeros(n_fourier)
    
    # Create and run MALA sampler
    sampler = AdaptiveMetropolisHastings(
        target= target,
        proposal = prop,
        initial_value = initial_state,
        adaptation_interval = 10,
        min_samples_adapt = 500,
        max_samples_adapt = 30000
    )
    
    sampler(n_samples)
    
    # Compute diagnostics
    diagnostics = MCMCDiagnostics(sampler, true_value=None)
    
    return sampler, diagnostics, heat_model
from typing import Tuple

class HeatDiffusionAnalysis:
    def __init__(self, sampler, model, true_coefficients, burn_in: int = 1000):
        self.sampler = sampler
        self.model = model
        self.true_coefficients = true_coefficients
        self.chain = sampler.chain[burn_in:]
        self.means = np.mean(self.chain, axis=0)
        
    def plot_coefficient_comparison(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot comparison of true vs estimated Fourier coefficients"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar plot comparison
        x = np.arange(len(self.true_coefficients))
        width = 0.35
        
        ax1.bar(x - width/2, self.true_coefficients, width, label='True')
        ax1.bar(x + width/2, self.means, width, label='Estimated')
        ax1.set_xlabel('Coefficient Index')
        ax1.set_ylabel('Value')
        ax1.set_title('True vs Estimated Fourier Coefficients')
        ax1.legend()
        
        # Add error bars using percentiles
        percentiles = np.percentile(self.chain, [2.5, 97.5], axis=0)
        errors = [self.means - percentiles[0], percentiles[1] - self.means]
        ax1.errorbar(x + width/2, self.means, yerr=errors, fmt='none', color='black')
        
        # Scatter plot of true vs estimated
        ax2.scatter(self.true_coefficients, self.means)
        ax2.plot([min(self.true_coefficients), max(self.true_coefficients)],
                 [min(self.true_coefficients), max(self.true_coefficients)],
                 'r--', label='Perfect Match')
        ax2.set_xlabel('True Values')
        ax2.set_ylabel('Estimated Values')
        ax2.set_title('True vs Estimated (Scatter)')
        ax2.legend()
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_temperature_profiles(self, x_grid: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Plot comparison of temperature profiles"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # True temperature profile
        true_temp = self.model.fourier_solution(x_grid, self.model.t_obs, 
                                              self.true_coefficients)
        
        # Mean estimated profile
        mean_temp = self.model.fourier_solution(x_grid, self.model.t_obs, 
                                              self.means)
        
        # Compute credible intervals
        n_samples = min(1000, len(self.chain))  # Use subset for efficiency
        sample_idx = np.random.choice(len(self.chain), n_samples, replace=False)
        temp_samples = np.array([
            self.model.fourier_solution(x_grid, self.model.t_obs, coef)
            for coef in self.chain[sample_idx]
        ])
        
        ci_lower, ci_upper = np.percentile(temp_samples, [2.5, 97.5], axis=0)
        
        # Plot everything
        ax.plot(x_grid, true_temp, 'r-', label='True')
        ax.plot(x_grid, mean_temp, 'b-', label='Mean Estimate')
        ax.fill_between(x_grid, ci_lower, ci_upper, color='blue', alpha=0.2,
                       label='95% Credible Interval')
        
        # Plot observations
        ax.scatter(self.model.x_obs, self.model.y_obs, c='black', 
                  marker='o', label='Observations')
        
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Temperature')
        ax.set_title('Temperature Profile Comparison')
        ax.legend()
        
        return fig, ax
    
    def print_coefficient_summary(self):
        """Print summary statistics for the coefficients"""
        print("\nFourier Coefficient Summary:")
        print("=" * 50)
        print(f"{'Index':^6} {'True':^10} {'Mean':^10} {'Std':^10} {'95% CI':^20}")
        print("-" * 50)
        
        percentiles = np.percentile(self.chain, [2.5, 97.5], axis=0)
        stds = np.std(self.chain, axis=0)
        
        for i in range(len(self.true_coefficients)):
            ci_str = f"[{percentiles[0,i]:.3f}, {percentiles[1,i]:.3f}]"
            print(f"{i:^6d} {self.true_coefficients[i]:^10.3f} "
                  f"{self.means[i]:^10.3f} {stds[i]:^10.3f} {ci_str:^20}")
        
        # Compute and print relative errors
        rel_errors = np.abs(self.means - self.true_coefficients) / np.abs(self.true_coefficients)
        print("\nRelative Errors:")
        print("-" * 50)
        for i, rel_error in enumerate(rel_errors):
            print(f"Coefficient {i}: {rel_error:.2%}")

def true_initial_temperature(x):
    """
    Define the true initial temperature profile we want to recover.
    This could be any function - here we use a smooth bump.
    """
    return np.exp(-(x - 0.5)**2 / 0.02)

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate spatial points for observations
    x_obs = np.linspace(0, 1, 50)
    t_obs = 0.1  # Time at which we observe
    
    # Create true initial condition on a fine grid for plotting
    x_fine = np.linspace(0, 1, 200)
    true_initial = true_initial_temperature(x_fine)
    
    # Generate synthetic observations using fundamental solution
    heat_model = HeatDiffusionInverse(x_obs, t_obs, None, N=20)  # More coefficients for accuracy
    
    # Generate true evolved temperature at observation time
    y_true = heat_model.fundamental_solution(x_obs, t_obs, true_initial_temperature)
    
    # Add noise to create observations
    noise_level = 0.05
    noise = noise_level * np.random.randn(len(x_obs))
    y_obs = y_true + noise
    
    # Plot initial condition and observations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot true initial condition
    ax1.plot(x_fine, true_initial, 'r-', label='True Initial Temperature')
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Temperature')
    ax1.set_title('True Initial Temperature Profile')
    ax1.legend()
    
    # Plot evolved solution and observations
    ax2.plot(x_obs, y_true, 'r-', label='Evolved True Solution')
    ax2.scatter(x_obs, y_obs, c='black', marker='o', label='Noisy Observations')
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Temperature')
    ax2.set_title(f'Temperature at t = {t_obs}')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Run MCMC to recover initial condition
    sampler, diagnostics, model = run_heat_diffusion_mcmc(
        x_obs=x_obs,
        t_obs=t_obs,
        y_obs=y_obs,
        n_samples=100000,
        n_fourier=20  # Using more coefficients to better approximate initial condition
    )
    
    # Print MCMC diagnostics
    diagnostics.print_summary()
    
    # Get recovered initial temperature
    recovered_coefs = np.mean(sampler.chain[50000:], axis=0)  # Using second half of chain
    recovered_initial = model.fourier_solution(x_fine, 0, recovered_coefs)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, true_initial, 'r-', label='True Initial')
    plt.plot(x_fine, recovered_initial, 'b-', label='Recovered Initial')
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature')
    plt.title('True vs Recovered Initial Temperature')
    plt.legend()
    plt.show()
    
    # Plot evolved solutions
    plt.figure(figsize=(10, 6))
    plt.plot(x_obs, y_true, 'r-', label='True Evolved')
    plt.plot(x_obs, model.fourier_solution(x_obs, t_obs, recovered_coefs), 
            'b-', label='Recovered Evolved')
    plt.scatter(x_obs, y_obs, c='black', marker='o', label='Observations')
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature')
    plt.title(f'Temperature Profiles at t = {t_obs}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

if __name__ == "Jimmy":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    x_obs = np.linspace(0, 1, 50)
    t_obs = 0.1
    true_A = np.array([1.0, 0.5, 0.25, 0.125])  # True Fourier coefficients
    
    # Create synthetic observations
    heat_model = HeatDiffusionInverse(x_obs, t_obs, None, N=len(true_A))
    y_true = heat_model.fourier_solution(x_obs, t_obs, true_A)
    
    # Add noise to create observations
    noise_level = 0.1
    noise = noise_level * np.random.randn(len(x_obs))
    y_obs = y_true + noise
    
    # Plot true solution and noisy observations
    plt.figure(figsize=(10, 6))
    plt.plot(x_obs, y_true, 'r-', label='True Solution')
    plt.scatter(x_obs, y_obs, c='black', marker='o', label='Noisy Observations')
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature')
    plt.title('True Solution vs Noisy Observations')
    plt.legend()
    plt.show()
    
    # Run MCMC
    sampler, diagnostics, model = run_heat_diffusion_mcmc(
        x_obs=x_obs, t_obs=t_obs, y_obs=y_obs,
        n_samples=100000, n_fourier=len(true_A)
    )
    
    # Create analysis object
    analysis = HeatDiffusionAnalysis(sampler, model, true_A)
    
    # Generate plots
    analysis.plot_coefficient_comparison()
    analysis.plot_temperature_profiles(np.linspace(0, 1, 200))
    
    # Print summary
    analysis.print_coefficient_summary()
    
    plt.show()
