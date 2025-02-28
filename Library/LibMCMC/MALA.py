## Metropolis Adjusted Langevin Algorithm
import time
from typing import Callable, Dict, Optional

import numpy as np
import scipy as sp

from Diagnostics import MCMCDiagnostics
from Distributions import Proposal, TargetDistribution
from GradientUtils import get_gradient_function
from MetropolisHastings import MetropolisHastings


class MALAProposal(Proposal):
    """MALA-specific proposal distribution incorporating gradient information"""

    def __init__(
        self,
        proposal_distribution: sp.stats.rv_continuous,
        scale: float,
        gradient_func: Callable,
    ):
        """
        Initialize MALA proposal.

        Args:
            proposal_distribution: Base proposal distribution (should be normal)
            scale: Initial step size (epsilon) for the Langevin dynamics
            gradient_func: Function computing gradient of log target density
        """
        super().__init__(proposal_distribution, 1.0)  # Initialize base with unit scale
        self.gradient_func = gradient_func
        self.epsilon = scale  # Step size for Langevin dynamics

    def propose(self, current: np.ndarray) -> np.ndarray:
        """
        Generate proposal using Langevin dynamics:
        x* = x + (ε/2)∇log(π(x)) + √ε * z, where z ~ N(0,I)
        """
        grad_log_target = self.gradient_func(current)
        drift = 0.5 * self.epsilon * grad_log_target
        # Use the RNG class's __call__ method instead of rvs
        diffusion = np.sqrt(self.epsilon) * self.proposal(0, 1, size=len(current))
        return current + drift + diffusion

    def proposal_log_density(
        self, proposed: np.ndarray, current: np.ndarray
    ) -> np.float64:
        """
        Compute log q(x'|x) for the MALA proposal.
        This is Gaussian with mean μ(x) = x + (ε/2)∇log(π(x))
        and variance ε*I
        """
        grad_log_target = self.gradient_func(current)
        mean = current + 0.5 * self.epsilon * grad_log_target
        return np.sum(
            sp.stats.norm.logpdf(proposed, loc=mean, scale=np.sqrt(self.epsilon))
        )


class MALA(MetropolisHastings):
    """Metropolis Adjusted Langevin Algorithm implementation"""

    def __init__(
        self,
        target_distribution: TargetDistribution,
        initial_state: np.ndarray,
        step_size: float = 0.1,
        adaptation_interval: int = 50,
        gradient_method: str = "numerical",
    ):
        """
        Initialize MALA sampler.

        Args:
            target_distribution: Target distribution to sample from
            gradient_func: Function computing gradient of log target density
            initial_state: Starting point for the chain
            step_size: Initial Langevin step size (epsilon)
            adaptation_interval: How often to adapt step size
        """
        # Get gradient function
        gradient_func = get_gradient_function(target_distribution, gradient_method)

        # Initialize MALA-specific proposal
        proposal = MALAProposal(
            sp.stats.norm,  # Always use normal for MALA
            step_size,
            gradient_func,
        )

        super().__init__(target_distribution, proposal, initial_state)

        self.gradient_func = gradient_func
        self.adaptation_interval = adaptation_interval

    def adapt_step_size(self):
        """
        Adapt the step size to achieve optimal acceptance rate (0.57 for MALA).
        Uses Robbins-Monro stochastic approximation.
        """
        if self._index < 100:  # Wait for some samples
            return

        current_acceptance = self.acceptance_count / self._index
        target_acceptance = 0.57  # Optimal acceptance rate for MALA

        # Robbins-Monro update
        factor = min(0.01, 1.0 / np.sqrt(self._index))
        log_step_size = np.log(self.proposal_distribution.epsilon)
        log_step_size += factor * (current_acceptance - target_acceptance)

        self.proposal_distribution.epsilon = np.exp(log_step_size)

    def __call__(self, n: int):
        """Run MALA for n iterations with step size adaptation"""
        if self._index + n > self.max_size:
            new_max = max(self.max_size * 2, self._index + n)
            new_chain = np.empty((new_max, self.chain.shape[1]))
            new_chain[: self._index] = self.chain[: self._index]
            self.chain = new_chain
            new_acceptance_rates = np.empty(new_max)
            new_acceptance_rates[: self._index] = self.acceptance_rates[: self._index]
            self.acceptance_rates = new_acceptance_rates
            self.max_size = new_max

        for i in range(n):
            if i == 200:
                self.burn(199)

            current = self.chain[self._index - 1]
            proposed = self.proposal_distribution.propose(current)

            log_alpha = self.acceptance_ratio(current, proposed)

            if np.log(self.uniform_rng.uniform()) < log_alpha:
                self.chain[self._index] = proposed
                self.acceptance_count += 1
            else:
                self.chain[self._index] = current

            if self._index % self.adaptation_interval == 0:
                self.adapt_step_size()

            self.acceptance_rates[self._index] = self.acceptance_count / self._index
            self._index += 1


class AdaptiveMALA(MALA):
    """MALA with preconditioning matrix adaptation"""

    def __init__(
        self,
        target_distribution: TargetDistribution,
        gradient_func: Callable,
        initial_state: np.ndarray,
        step_size: float = 0.1,
        adaptation_interval: int = 50,
        min_samples_adapt: int = 500,
        max_samples_adapt: int = 1000,
    ):
        super().__init__(
            target_distribution,
            gradient_func,
            initial_state,
            step_size,
            adaptation_interval,
        )

        self.min_samples_adapt = min_samples_adapt
        self.max_samples_adapt = max_samples_adapt

        # Initialize preconditioning matrix estimate
        d = len(initial_state)
        self.empirical_mean = initial_state.copy()
        self.preconditioner = np.eye(d)

    def update_preconditioner(self, new_sample: np.ndarray):
        """Update preconditioning matrix using sample covariance"""
        n = self._index - self.min_samples_adapt
        if n <= 0:
            return

        old_mean = self.empirical_mean.copy()
        self.empirical_mean = (n * old_mean + new_sample) / (n + 1)

        # Update sample covariance
        cov_update = (n / (n + 1)) * self.preconditioner + (
            n / ((n + 1) ** 2)
        ) * np.outer(old_mean - new_sample, old_mean - new_sample)

        # Ensure numerical stability
        self.preconditioner = cov_update + 1e-6 * np.eye(len(new_sample))

    def __call__(self, n: int):
        """Run preconditioned MALA"""
        if self._index + n > self.max_size:
            new_max = max(self.max_size * 2, self._index + n)
            new_chain = np.empty((new_max, self.chain.shape[1]))
            new_chain[: self._index] = self.chain[: self._index]
            self.chain = new_chain
            new_acceptance_rates = np.empty(new_max)
            new_acceptance_rates[: self._index] = self.acceptance_rates[: self._index]
            self.acceptance_rates = new_acceptance_rates
            self.max_size = new_max

        for i in range(n):
            if i == 200:
                self.burn(199)

            current = self.chain[self._index - 1]
            proposed = self.proposal_distribution.propose(current)

            log_alpha = self.acceptance_ratio(current, proposed)

            if np.log(self.uniform_rng.uniform()) < log_alpha:
                accepted_state = proposed
                self.chain[self._index] = proposed
                self.acceptance_count += 1
            else:
                accepted_state = current
                self.chain[self._index] = current

            if self._index % self.adaptation_interval == 0:
                self.adapt_step_size()
                if self._index >= self.min_samples_adapt:
                    self.update_preconditioner(accepted_state)

            self.acceptance_rates[self._index] = self.acceptance_count / self._index
            self._index += 1


class GaussianTarget(TargetDistribution):
    def __init__(self):
        prior = sp.stats.norm
        likelihood = sp.stats.norm
        # 2D target - generating two parameters
        data = sp.stats.norm.rvs(1, 1, 1000)

        super().__init__(prior, likelihood, data, 1.0)


# Example usage:
if __name__ == "__main__":
    target = GaussianTarget()
    initial_state = np.array([0.0])

    # Create MALA sampler
    sampler = MALA(
        target_distribution=target,
        initial_state=initial_state,
        step_size=0.0030,
        gradient_method="numerical",
    )

    sampler(10000)
    proposed = sampler.proposal_distribution.propose(initial_state)
    print(f"Proposed state: {proposed}")
    print(f"Acceptance ratio: {sampler.acceptance_ratio(initial_state, proposed)}")
    diagnostics = MCMCDiagnostics(sampler, true_value=target.data)

    diagnostics.print_summary()
