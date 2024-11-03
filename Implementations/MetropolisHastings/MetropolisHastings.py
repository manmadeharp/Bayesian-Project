import scipy as sp
import numpy as np
from numpy.random import Generator, Philox, SeedSequence
from PRNG import SEED
from Distributions import Proposal, TargetDistribution

# Default Values
# - Seed Value


class MetropolisHastings:
    def __init__(
        self,
        target_distribution: TargetDistribution,
        proposal_distribution: Proposal,
        initialstate,
    ):
        """
        Initialize the Metropolis-Hastings algorithm.
        :param target_distribution:
        :param proposal_distribution:
        :param initialstate:
        """
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution

        self.max_size = 10000  # Or some reasonable default
        self.chain = np.empty((self.max_size, len(initialstate)))
        self.chain[0] = initialstate
        self._index = 1
        self.uniform_rng = np.random.default_rng(
            seed=SEED
        )  # Using Philox for Reproducability

    def acceptance_ratio(self, current: np.ndarray, proposed: np.ndarray) -> np.float64:
        """
        Acceptance ratio for the Metropolis-Hastings algorithm.
        :param current: Current state of the chain.
        :param proposed: Proposed state of the chain.
        :return: Acceptance ratio.
        """

        prior_ratio = self.target_distribution.log_prior(
            proposed
        ) - self.target_distribution.log_prior(current)
        likelihood_ratio = self.target_distribution.log_likelihood(
            proposed
        ) - self.target_distribution.log_likelihood(current)
        transition_ratio = self.proposal_distribution.proposal_log_density(
            current, proposed
        ) - self.proposal_distribution.proposal_log_density(
            proposed, current
        )  # Previous given new over new given previous

        log_ratio = prior_ratio + likelihood_ratio + transition_ratio
        return min(np.float64(0), log_ratio)

    def __call__(self, n: int):
        """
        Run the Metropolis-Hastings algorithm for n iterations.
        :param n:
        """
        # Resize if needed
        if self._index + n > self.max_size:
            new_max = max(self.max_size * 2, self._index + n)
            new_chain = np.empty((new_max, self.chain.shape[1]))
            new_chain[: self._index] = self.chain[: self._index]
            self.chain = new_chain
            self.max_size = new_max

        for i in range(n):
            current = self.chain[self._index - 1]
            proposed = self.proposal_distribution.propose(current)
            if np.log(self.uniform_rng.uniform()) < self.acceptance_ratio(
                current, proposed
            ):
                self.chain[self._index] = proposed
            else:
                self.chain[self._index] = current
            self._index += 1


class AdaptiveMetropolisHastings(MetropolisHastings):
    def __init__(
        self,
        target: TargetDistribution,
        proposal: Proposal,
        initial_value,
        adaptation_interval: int = 100,
        target_acceptance: float = 0.234,
        adaptation_scale: float = 2.4,
    ):
        super().__init__(target, proposal, initial_value)
        self.acceptance_count = 0
        self.adaptation_interval = adaptation_interval
        self.target_acceptance = target_acceptance
        self.adaptation_scale = adaptation_scale * 2 / initial_value.size

    def update_proposal(self):
        """Update proposal covariance based on chain history"""
        chain_segment = self.chain[: self._index]

        cov = np.cov(chain_segment.T)
        scaled_cov = (
            self.acceptance_count / self.chain.size
        ) * self.adaptation_scale * cov + np.eye(cov.shape[0]) * 1e-6
        self.proposal_distribution.beta = sp.stats.Covariance.from_cholesky(scaled_cov)

    def __call__(self, n: int):
        """Run adaptive MCMC"""
        # Resize if needed
        if self._index + n > self.max_size:
            new_max = max(self.max_size * 2, self._index + n)
            new_chain = np.empty((new_max, self.chain.shape[1]))
            new_chain[: self._index] = self.chain[: self._index]
            self.chain = new_chain
            self.max_size = new_max

        for i in range(n):
            current = self.chain[self._index - 1]
            proposed = self.proposal_distribution.propose(current)
            if np.log(self.uniform_rng.uniform()) < self.acceptance_ratio(
                current, proposed
            ):
                self.chain[self._index] = proposed
                self.acceptance_count += 1
            else:
                self.chain[self._index] = current

            if (
                self._index % self.adaptation_interval == 0
            ):  # Adapt every 100 iterations
                self.update_proposal()

            self._index += 1
