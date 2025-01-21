import sys
import os
import numpy as np
import scipy.stats as sp
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BayesianInference.Distributions import Proposal, TargetDistribution
from BayesianInference.MetropolisHastings import MetropolisHastings, AdaptiveMetropolisHastings
from BayesianInference.Diagnostics import MCMCDiagnostics

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_double_well_distribution(sigma=0.3, resolution=100):
    """
    Create comprehensive visualizations of the double-well potential
    and its corresponding probability distribution.
    """
    # Create grid of points
    x = np.linspace(-2, 2, resolution)
    y = np.linspace(-2, 2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute potential energy
    V = (X**2 - 1)**2/(4*sigma**2) + (Y**2 - 1)**2/(4*sigma**2)
    
    # Compute probability density (unnormalized)
    P = np.exp(-V)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Potential Energy Surface (3D)
    ax1 = fig.add_subplot(231, projection='3d')
    surf1 = ax1.plot_surface(X, Y, V, cmap=cm.viridis,
                            linewidth=0, antialiased=True)
    ax1.set_title('Potential Energy Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('V(x,y)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # 2. Potential Energy Contours
    ax2 = fig.add_subplot(232)
    contour1 = ax2.contour(X, Y, V, levels=20, cmap='viridis')
    ax2.clabel(contour1, inline=True, fontsize=8)
    ax2.set_title('Potential Energy Contours')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # 3. Probability Density Surface (3D)
    ax3 = fig.add_subplot(233, projection='3d')
    surf2 = ax3.plot_surface(X, Y, P, cmap=cm.viridis,
                            linewidth=0, antialiased=True)
    ax3.set_title('Density Surface')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('p(x,y)')
    fig.colorbar(surf2, ax=ax3, shrink=0.5, aspect=5)
    
    # 4. Probability Density Contours
    ax4 = fig.add_subplot(234)
    contour2 = ax4.contour(X, Y, P, levels=20, cmap='viridis')
    ax4.clabel(contour2, inline=True, fontsize=8)
    ax4.set_title('Density Contours')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    
    # 5. Cross-sections along X and Y
    ax5 = fig.add_subplot(235)
    ax5.plot(x, np.exp(-(x**2 - 1)**2/(4*sigma**2)), 'b-', 
             label='X cross-section (Y=0)')
    ax5.plot(y, np.exp(-(y**2 - 1)**2/(4*sigma**2)), 'r--', 
             label='Y cross-section (X=0)')
    ax5.set_title('Density Cross-sections')
    ax5.set_xlabel('Position')
    ax5.set_ylabel('Density')
    ax5.legend()
    
    # 6. Log Probability Density
    ax6 = fig.add_subplot(236)
    log_p = np.log(P)
    contour3 = ax6.contour(X, Y, log_p, levels=20, cmap='viridis')
    ax6.clabel(contour3, inline=True, fontsize=8)
    ax6.set_title('Log Density Contours')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_mcmc_results(sampler, name, sigma=0.3, resolution=100):
    """
    Plot MCMC samples overlaid on the true distribution
    """
    # Create grid of points for contour
    x = np.linspace(-2, 2, resolution)
    y = np.linspace(-2, 2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute potential energy and exponential
    V = (X**2 - 1)**2/(4*sigma**2) + (Y**2 - 1)**2/(4*sigma**2)
    P = np.exp(-V)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First subplot - contour plot with samples (keeping this as is)
    contour1 = ax1.contour(X, Y, P, levels=20, cmap='viridis', alpha=0.7)
    ax1.clabel(contour1, inline=True, fontsize=8)
    
    # Get chain and handle burn-in
    if hasattr(sampler, 'min_samples_adapt'):
        chain = sampler.chain[sampler.min_samples_adapt:sampler._index]
    else:
        burn_in = int(0.2 * len(sampler.chain))
        chain = sampler.chain[burn_in:sampler._index]

    
    n_zeros = np.sum(np.all(np.abs(chain) < 0.01, axis=1))
    print(f"Samples near (0,0):", n_zeros)
    
    # Plot MCMC samples on contour
    ax1.scatter(chain[:,0], chain[:,1], c='purple', alpha=0.7, s=1, 
                label='MCMC samples')
    ax1.set_title('MCMC Samples vs True Distribution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    
    # Second subplot - 3D histogram with empirical surface
    ax3d = fig.add_subplot(122, projection='3d')
    
    # Plot the true surface
    surf = ax3d.plot_surface(X, Y, P, cmap='viridis', alpha=0.5)
    
    # Create histogram of samples
    hist, xedges, yedges = np.histogram2d(chain[:,0], chain[:,1], 
                                         bins=100, 
                                         range=[[-2, 2], [-2, 2]])
    
    # Create bar positions
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = hist.flatten()
    
    # Scale histogram to match surface
    scale_factor = np.max(P) / np.max(dz)
    dz = dz * scale_factor
    
    # Plot histogram bars
    ax3d.bar3d(xpos, ypos, zpos, dx, dy, dz, color='purple', alpha=0.3)
    ax3d.set_title('Empirical vs Sample Distribution')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()
    
    fig.savefig(f"./Plots/Comparison_plots_{name}")
    return fig

# Function to create animated trace of MCMC samples
def create_mcmc_animation(sampler, sigma=0.3, resolution=100, 
                         n_frames=200, samples_per_frame=50):
    """
    Create animation of MCMC sampling process.
    Returns list of figures that can be converted to GIF.
    """
    # Create grid for contour
    x = np.linspace(-2, 2, resolution)
    y = np.linspace(-2, 2, resolution)
    X, Y = np.meshgrid(x, y)
    V = (X**2 - 1)**2/(4*sigma**2) + (Y**2 - 1)**2/(4*sigma**2)
    P = np.exp(-V)
    
    figures = []
    chain = sampler.chain[sampler.min_samples_adapt:]
    
    for i in range(n_frames):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot probability density contours
        ax.contour(X, Y, P, levels=20, cmap='viridis', alpha=0.5)
        
        # Plot MCMC samples up to current frame
        current_samples = chain[:(i+1)*samples_per_frame]
        ax.scatter(current_samples[:,0], current_samples[:,1], 
                  c='red', alpha=0.1, s=1)
        
        ax.set_title(f'MCMC Samples (n={len(current_samples)})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        figures.append(fig)
        plt.close(fig)
    
    return figures

class DoubleWellTarget(TargetDistribution):
    """Bimodal target using double-well potential"""
    def __init__(self, sigma: float = 0.3):
        self.sigma = sigma
        
        # Setup prior and likelihood for double-well structure
        prior = sp.multivariate_normal(mean=np.zeros(2), cov=np.eye(2))
        likelihood = sp.multivariate_normal
        data = None  # No actual data needed for this test case
        
        super().__init__(prior, likelihood, data, sigma)
        
        # Store analytical values
        self._analytical_mean = np.array([0.0, 0.0])
        self._analytical_var = np.array([[1.0, 0.0], [0.0, 1.0]]) * self.sigma**2

    def log_likelihood(self, x: np.ndarray) -> np.float64:
        """Double-well potential in first dimension"""
        return np.float64(-0.25/(self.sigma**2) * (x[0]**2 - 1)**2)

    def log_prior(self, x: np.ndarray) -> np.float64:
        """Double-well potential in second dimension"""
        return np.float64(-0.25/(self.sigma**2) * (x[1]**2 - 1)**2)

    def analytical_mean(self) -> np.ndarray:
        """True mean by symmetry"""
        return self._analytical_mean

    def analytical_covariance(self) -> np.ndarray:
        """True covariance by symmetry and numerical integration"""
        return self._analytical_var

def setup_samplers(
    target: TargetDistribution,
    initial_state: np.ndarray,
    mh_scale: float = 0.3
) -> Tuple[MetropolisHastings, AdaptiveMetropolisHastings]:
    """Setup both MH and Adaptive MH samplers with proper parameters"""
    
    # Standard MH with fixed proposal
    mh_proposal = Proposal(sp.multivariate_normal, scale=mh_scale*np.eye(2))
    mh_sampler = MetropolisHastings(target, mh_proposal, initial_state)
    
    # Adaptive MH with proper parameters
    amh_proposal = Proposal(sp.multivariate_normal, scale=np.eye(2))
    amh_sampler = AdaptiveMetropolisHastings(
        target=target,
        proposal=amh_proposal,
        initial_value=initial_state,
        adaptation_interval=5,
        target_acceptance=0.234,
        adaptation_scale=2.4,
        min_samples_adapt=4000,
        max_samples_adapt=20000
    )
    
    return mh_sampler, amh_sampler

def run_comparison(n_samples: int = 100000) -> Tuple[MetropolisHastings, AdaptiveMetropolisHastings, TargetDistribution]:
    """Run and compare MCMC samplers on bimodal target"""
    
    # Validate inputs
    if n_samples < 5000:
        raise ValueError("n_samples should be at least 5000 for reliable results")
    
    # Setup target and samplers
    target = DoubleWellTarget(sigma=0.3)
    initial_state = np.zeros(2)
    mh_sampler, amh_sampler = setup_samplers(target, initial_state)
    
    # Run samplers
    print("Running Standard MH...")
    mh_sampler(n_samples)
    print("Running Adaptive MH...")
    amh_sampler(n_samples)
    
    # Analyze results
    print("\nAnalytical Values:")
    print(f"Mean: {target.analytical_mean()}")
    print(f"Covariance:\n{target.analytical_covariance()}")
    
    print("\nStandard MH Diagnostics:")
    mh_diag = MCMCDiagnostics(mh_sampler, target.analytical_mean())
    mh_diag.plot_diagnostics()
    #mh_diag.plot_target_distribution()
    mh_diag.print_summary()
    
    print("\nAdaptive MH Diagnostics:")
    amh_diag = MCMCDiagnostics(amh_sampler, target.analytical_mean())
    amh_diag.plot_diagnostics()
    #amh_diag.plot_target_distribution()
    amh_diag.print_summary()
    
    return mh_sampler, amh_sampler, target

if __name__ == "__main__":
    mh, amh, target = run_comparison()

    print(dir(mh.__class__))

    # Plot the true distribution
    plot_double_well_distribution(sigma=0.3)
    
    # Plot MCMC results
    plot_mcmc_results(mh, sigma=0.3, name="Metropolis")
    plot_mcmc_results(amh, sigma=0.3, name="Adaptive")  # Using adaptive MH results
