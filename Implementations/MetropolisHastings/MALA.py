import numpy as np
import scipy as sp
from typing import Callable, Optional
from MetropolisHastings import MetropolisHastings
from Distributions import Proposal, TargetDistribution

## Metropolis Adjusted Langevin Algorithm WIP

class GradientComputer:
    """
    Handles computation of gradients for the target distribution.
    Can use either numerical or analytical gradients.
    """
    def __init__(self, target: TargetDistribution, eps: float = 1e-8):
        self.target = target
        self.eps = eps
        
    def numerical_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient using finite differences"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += self.eps
            x_minus = x.copy()
            x_minus[i] -= self.eps
            
            # Compute gradient of log posterior
            grad[i] = (
                (self.target.log_prior(x_plus) + 
                 self.target.log_likelihood(x_plus) -
                 self.target.log_prior(x_minus) - 
                 self.target.log_likelihood(x_minus)) / (2 * self.eps)
            )
        return grad

class LangevinProposal(Proposal):
    """
    Handles Langevin dynamics-based proposals.
    """
    def __init__(
        self,
        proposal_distribution: sp.stats.rv_continuous,
        step_size: float,
        gradient_computer: GradientComputer
    ):
        super().__init__(proposal_distribution, np.sqrt(step_size))
        self.step_size = step_size
        self.gradient_computer = gradient_computer
        
    def propose(self, current: np.ndarray) -> np.ndarray:
        """Generate proposal using Langevin dynamics"""
        # Compute gradient-based drift
        gradient = self.gradient_computer.numerical_gradient(current)
        mean = current + 0.5 * self.step_size * gradient
        
        # Add noise scaled by sqrt(step_size)
        noise = np.sqrt(self.step_size) * self.proposal(
            np.zeros_like(current),
            np.eye(len(current))
        )
        return mean + noise
        
    def proposal_log_density(self, proposed: np.ndarray, current: np.ndarray) -> np.float64:
        """Compute log density of the Langevin proposal"""
        # Compute means for forward and backward proposals
        forward_gradient = self.gradient_computer.numerical_gradient(current)
        forward_mean = current + 0.5 * self.step_size * forward_gradient
        
        # Use parent class proposal distribution for density computation
        return self.proposal_distribution.logpdf(
            proposed,
            forward_mean,
            np.sqrt(self.step_size) * np.eye(len(current))
        )

class MALA(MetropolisHastings):
    """
    Metropolis-Adjusted Langevin Algorithm
    Uses gradient information for intelligent proposals
    """
    def __init__(
        self,
        target: TargetDistribution,
        step_size: float,
        initial_state: np.ndarray,
    ):
        # Set up gradient computation
        self.gradient_computer = GradientComputer(target)
        
        # Set up Langevin proposal mechanism
        proposal = LangevinProposal(
            sp.stats.multivariate_normal,
            step_size,
            self.gradient_computer
        )
        
        # Initialize parent class
        super().__init__(target, proposal, initial_state)
        
    def acceptance_ratio(self, current: np.ndarray, proposed: np.ndarray) -> np.float64:
        """
        Compute acceptance ratio accounting for asymmetric proposals
        """
        # Standard MH ratio terms
        prior_ratio = (self.target_distribution.log_prior(proposed) - 
                      self.target_distribution.log_prior(current))
        likelihood_ratio = (self.target_distribution.log_likelihood(proposed) - 
                          self.target_distribution.log_likelihood(current))
        
        # Proposal ratio (forward vs backward proposals)
        proposal_ratio = (
            self.proposal_distribution.proposal_log_density(current, proposed) -
            self.proposal_distribution.proposal_log_density(proposed, current)
        )
        
        return min(0.0, prior_ratio + likelihood_ratio + proposal_ratio)

# Example usage with diagnostics
def run_mala_with_diagnostics(
    target: TargetDistribution,
    initial_state: np.ndarray,
    step_size: float,
    n_iterations: int
) -> dict:
    """
    Run MALA algorithm and return diagnostics
    """
    # Initialize and run sampler
    sampler = MALA(target, step_size, initial_state)
    sampler(n_iterations)
    
    # Compute diagnostics
    acceptance_rate = sampler.acceptance_count / n_iterations
    
    return {
        'chain': sampler.chain[:sampler._index],
        'acceptance_rate': acceptance_rate,
        'step_size': step_size
    }           

