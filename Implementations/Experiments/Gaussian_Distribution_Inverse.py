from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
from scipy.fft import fft, fftfreq
from scipy.integrate import dblquad
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BayesianInference.Distributions import Proposal, TargetDistribution
from BayesianInference.MetropolisHastings import AdaptiveMetropolisHastingsVOne, AdaptiveMetropolisHastingsVTwo, MetropolisHastings
from BayesianInference.Diagnostics import MCMCDiagnostics  # Using our new diagnostics file!

class MultivariateGaussianInverseTarget(TargetDistribution):
    def __init__(self, data, mean: np.ndarray, noise_sigma, prior_mean, prior_cov):
        """

        Mean is used for data generation 
        """
        self.noise_sigma = noise_sigma
        
        self.prior = sp.multivariate_normal(prior_mean, prior_cov)
        self.likelihood = sp.multivariate_normal
        super().__init__(self.prior, self.likelihood, data, noise_sigma)


    def log_likelihood(self, x: np.ndarray) -> np.float64:
        """Log likelihood of data given parameters theta"""
        return super().log_likelihood(x)

    def log_prior(self, x: np.ndarray) -> np.float64:
        """Log prior on parameters"""
        t = super().log_prior(x)
        return t
class BananaTarget(TargetDistribution):
    """Banana-shaped distribution from Haario et al."""
    def __init__(self, b=0.1):
        self.b = b
        self.mean = np.array([0, -100*b])
        self.cov = np.array([[1, 2*b], [2*b, 1 + 4*b**2]])
        
    def log_likelihood(self, x: np.ndarray) -> np.float64:
        return self.log_prior(x)
    
    def log_prior(self, x: np.ndarray) -> np.float64:
        x1, x2 = x[0], x[1] + self.b * x[0]**2 - 100*self.b
        return np.float64(-0.5 * (x1**2 + x2**2))


def run_inverse_example(n_samples: int = 10000):
    # Generate synthetic data from true parameter values
    mean = np.array([10.0, -0.2]) 
    sigma = np.array([[5.00, 2.5], 
                      [2.5, 5.00]])
    #noise = sp.norm.rvs(0, noise_sigma, size=2)
    #data = true_params + noise

    data = np.random.multivariate_normal(mean, sigma, 500)
    empirical_mean = np.mean(data, axis=0)
    empirical_cov = np.cov(data.T)

    # Setup target with generated data
    target = MultivariateGaussianInverseTarget(data, mean, sigma, mean, sigma)

    # Setup sampler
    initial_state = np.zeros(2)
    proposal = Proposal(sp.multivariate_normal, scale=np.eye(2) )
    
    sampler = AdaptiveMetropolisHastingsVOne(
        target=target,
        proposal=proposal,
        initial_value=initial_state,
        min_samples_adapt=500,
        max_samples_adapt=1000

    )

    sampleTwo = AdaptiveMetropolisHastingsVTwo(
        target=target,
        proposal=proposal,
        initial_value=initial_state,
        min_samples_adapt=500,
        max_samples_adapt=1000
    )

    sampleThree = MetropolisHastings(
        target_distribution = target,
        proposal_distribution = Proposal(sp.multivariate_normal, 0.002*sigma),
        initialstate = initial_state,
    )


    # Use our diagnostics class!

    


    sampler(n_samples)
    #    diagnostics = MCMCDiagnostics(sampler)
    #    diagnostics.plot_diagnostics()
    #    #diagnostics.plot_target_distribution()
    #    diagnostics.print_summary()
    #
    sampleTwo(n_samples)
    #    diagnostics = MCMCDiagnostics(sampleTwo)
    #    diagnostics.plot_diagnostics()
    #    #diagnostics.plot_target_distribution()
    #    diagnostics.print_summary()

    sampleThree(n_samples)
    diagnostics = MCMCDiagnostics(sampleThree)
    diagnostics.plot_diagnostics()
    #diagnostics.plot_target_distribution()
    diagnostics.print_summary()
    print(f'True Mean: {mean}, True Cov: {sigma}')



if __name__ == "__main__":
    run_inverse_example()


