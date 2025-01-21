import numpy as np
import scipy as sp

from Distributions import Proposal, TargetDistribution
from PRNG import SEED

# Default Values
# - Seed Value
# SEED = 112


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
        self.chain = np.zeros((self.max_size, len(initialstate)))
        self.chain[0] = initialstate
        self._index = 1
        self.uniform_rng = np.random.default_rng(
            seed=SEED
        )  # Using Philox for Reproducability

        self.acceptance_count = 0
        self.acceptance_rates = np.zeros(self.max_size)

        self.burnt = False

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
        assert np.isscalar(prior_ratio)
        assert np.isscalar(likelihood_ratio)
        assert np.isscalar(transition_ratio)

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
            new_acceptance_rates = np.empty(new_max)
            new_acceptance_rates[: self._index] = self.acceptance_rates[: self._index]
            self.acceptance_rates = new_acceptance_rates

        for i in range(n):
            if i == 200:
                self.burn(199)
            current = self.chain[self._index - 1]
            proposed = self.proposal_distribution.propose(current)
            if np.log(self.uniform_rng.uniform()) < self.acceptance_ratio(
                current, proposed
            ):
                self.chain[self._index] = proposed
                self.acceptance_count += 1
            else:
                self.chain[self._index] = current

            self.acceptance_rates[self._index] = self.acceptance_count / self._index
            self._index += 1

    def burn(self, n: int):
        if n < self._index:
            # Keep only samples after burn point
            burned_chain = self.chain[n : self._index]

            # Create new arrays and copy burned data
            new_chain = np.empty((self.max_size, self.chain.shape[1]))
            new_chain[: len(burned_chain)] = burned_chain

            # Reset acceptance tracking
            self.acceptance_count = 0
            self.acceptance_rates = np.zeros(self.max_size)

            # Update chain and index
            self.chain = new_chain
            self._index = len(burned_chain)
            print("burnt is: ", self.burnt)
            self.burnt = True

        else:
            raise ValueError("Burn-in exceeds the number of samples in the chain.")

    #    def burn(self, n):
    #        print("chain delte: ", len(self.chain))
    #        self.chain = self.chain[n:]
    #       print("chain deleteafter: ", len(self.chain))

    def writeToFile(self, filename: str) -> None:
        """
        Write the MCMC chain to a CSV file.

        Args:
            filename: Path to the CSV file
        """
        import numpy as np

        # Get the actual chain data (exclude empty rows)
        chain_data = self.chain[: self._index]

        # Write using numpy's savetxt
        np.savetxt(filename, chain_data, delimiter=",", fmt="%.8f")


class AdaptiveMetropolisHastingsVOne(MetropolisHastings):
    def __init__(
        self,
        target: TargetDistribution,
        proposal: Proposal,
        initial_value: np.ndarray,
        adaptation_interval: int = 50,
        min_samples_adapt: int = 500,
        max_samples_adapt: int = 1000,
    ):
        # Use parent class initialization
        super().__init__(target, proposal, initial_value)

        # Add only what's needed for adaptation
        self.adaptation_interval = adaptation_interval
        self.min_samples_adapt = min_samples_adapt
        self.max_samples_adapt = max_samples_adapt

        # Setup for covariance adaptation
        d = len(initial_value)
        self.scale = (2.38**2) / d
        self.all_samples = np.zeros((max_samples_adapt, d))
        self.all_samples[0] = initial_value
        self.n_samples = 1

    def update_covariance(self, state: np.ndarray):
        """Update covariance estimate with new state"""
        if self.n_samples >= self.max_samples_adapt:
            return

        self.all_samples[self.n_samples] = state

        if self.n_samples > 1:
            # Update proposal covariance using all samples
            sample_cov = np.cov(self.all_samples[: self.n_samples + 1].T)
            reg = 1e-6 * np.diag(np.diag(sample_cov))
            self.proposal_distribution.beta = self.scale * (sample_cov + reg)

        self.n_samples += 1

    def __call__(self, n: int):
        # Use parent's chain management
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

            # Use parent's acceptance ratio
            log_alpha = self.acceptance_ratio(current, proposed)

            if np.log(self.uniform_rng.uniform()) < log_alpha:
                self.chain[self._index] = proposed
                self.acceptance_count += 1
            else:
                self.chain[self._index] = current

            # Update covariance using current state
            if self._index % self.adaptation_interval == 0:
                self.update_covariance(self.chain[self._index])

            self.acceptance_rates[self._index] = self.acceptance_count / self._index
            self._index += 1


class AdaptiveMetropolisHastings(MetropolisHastings):
    def __init__(
        self,
        target: TargetDistribution,
        proposal: Proposal,
        initial_value: np.ndarray,
        adaptation_interval: int = 1,  # Changed to 1 based on Haario et al.
        target_acceptance: float = 0.234,  # Roberts & Rosenthal optimal
        adaptation_scale: float = 2.4,  # Haario et al. optimal
        min_samples_adapt: int = 800,  # Earlier adaptation from Roberts & Rosenthal
        max_samples_adapt: int = 1500,
    ):
        super().__init__(target, proposal, initial_value)
        self.adaptation_interval = adaptation_interval
        self.target_acceptance = target_acceptance

        # Optimal scaling from Haario et al.
        d = len(initial_value)
        self.adaptation_scale = (adaptation_scale**2) / d

        self.min_samples_adapt = min_samples_adapt
        self.max_samples_adapt = max_samples_adapt

        # Initialize empirical mean and covariance (Haario et al.)
        self.empirical_mean = initial_value.copy()
        self.empirical_cov = np.eye(d)
        self.n_samples = 1

    def update_empirical_estimates(self, new_sample: np.ndarray):
        """Update running covariance estimate using R&R 2009 method"""
        n = self._index - self.min_samples_adapt
        if n <= 0:
            return

        # Update mean
        old_mean = self.empirical_mean.copy()
        self.empirical_mean = (n * old_mean + new_sample) / (n + 1)

        # Update covariance using R&R formula
        self.empirical_cov = (n / (n + 1)) * self.empirical_cov + (
            n / ((n + 1) ** 2)
        ) * np.outer(old_mean - new_sample, old_mean - new_sample)

    def get_adaptation_weight(self) -> float:
        """Roberts & Rosenthal 2009 adaptation rate"""
        if self._index <= self.min_samples_adapt:
            return 1.0
        elif self._index >= self.max_samples_adapt:
            return 0.0

        t = self._index - self.min_samples_adapt
        # This gives rate decaying like 1/t
        return min(1.0, self.adaptation_scale / t)

    def update_proposal(self):
        """Update proposal covariance using R&R scheme"""
        if self._index < self.min_samples_adapt:
            return
        if self._index == self.min_samples_adapt:
            print(self.proposal_distribution.beta)
            return

        if self._index > self.max_samples_adapt:
            return
        if self._index == self.max_samples_adapt:
            print(self.proposal_distribution.beta)
            return

        scaled_cov = self.empirical_cov
        scaled_cov += 1e-6 * np.eye(self.empirical_cov.shape[0])

        self.proposal_distribution.beta = scaled_cov

    def __call__(self, n: int):
        """Run adaptive MCMC with online updates"""
        if self._index + n > self.max_size:
            new_max = max(self.max_size * 2, self._index + n)
            new_chain = np.empty((new_max, self.chain.shape[1]))
            new_chain[: self._index] = self.chain[: self._index]
            self.chain = new_chain
            self.max_size = new_max
            new_acceptance_rates = np.empty(new_max)
            new_acceptance_rates[: self._index] = self.acceptance_rates[: self._index]
            self.acceptance_rates = new_acceptance_rates
            self.max_size = new_max

        for i in range(n):
            if (self._index == self.min_samples_adapt // 2) and not self.burnt:
                print("Burning")
                self.burn(self.min_samples_adapt // 2 - 1)
            current = self.chain[self._index - 1]
            proposed = self.proposal_distribution.propose(current)

            log_alpha = self.acceptance_ratio(current, proposed)

            if np.log(self.uniform_rng.uniform()) < log_alpha:
                accepted_state = proposed
                self.chain[self._index] = proposed
                self.acceptance_count += 1
                # Update empirical estimates only on acceptance
            else:
                accepted_state = current
                self.chain[self._index] = current

            if (
                self._index % self.adaptation_interval == 0
                and self._index > self.min_samples_adapt
            ):
                self.update_empirical_estimates(accepted_state)
                self.update_proposal()

            self.acceptance_rates[self._index] = self.acceptance_count / self._index
            self._index += 1
