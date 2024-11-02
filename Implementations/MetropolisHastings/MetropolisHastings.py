import scipy as sp
import numpy as np
from numpy.random import Generator, Philox, SeedSequence
from numpy.typing import NDArray
from typing import Optional
# Default Values
# - Seed Value
SEED = None

class RNG:
    def __init__(self, seed: Optional[int], distribution: sp.stats.rv_continuous):
        self.ss = SeedSequence(seed)
        self.rg = Generator(Philox(self.ss)) # Use Philox for parallel applications
        self.rng_distribution = distribution
    def __call__(self, loc, scale, size: Optional[int] = None, *args, **kwargs):
        return self.rng_distribution.rvs(loc, scale, size=size, random_state=self.rg,
                                         *args, **kwargs)


class Proposal:
    def __init__(self,
                 proposal_distribution: sp.stats.rv_continuous,
                 scale: np.ndarray):
        self.proposal_distribution = proposal_distribution
        self.proposal = RNG(SEED, proposal_distribution)
        if np.isscalar(scale):
            self.beta = np.sqrt(scale)
        else:
            self.beta = sp.stats.Covariance.from_cholesky(scale) # L*x ~ N(0, Sigma)

    def propose(self, current: np.ndarray) -> np.ndarray:
        return self.proposal(current, self.beta)

    def proposal_log_density(self, state: np.ndarray, loc: np.ndarray, ) -> np.ndarray:
        return self.proposal_distribution.logpdf(state, loc, self.beta)

# Test Proposal
test = Proposal(sp.stats.multivariate_normal, np.array([[1, 2], [2, 1]]))
print(test.propose(np.array([1.0, 12])))
print(test.proposal_log_density(np.array([1.0, 12]), np.array([1.0, 12])))

class TargetDistribution:
    def __init__(self, prior: sp.stats.rv_continuous,
                 likelihood: sp.stats.rv_continuous,
                 data,
                 sigma: float):
        self.prior = prior
        self.likelihood = likelihood
        # likelihood
        self.data = data
        self.data_sigma = sigma

    def log_likelihood(self, x: np.ndarray) -> np.float64:
        """
        Likelihood of our data given the parameters x.
        I.E the distribution of the data given the parameters x.
        :param x:
        :return:
        """
        return np.sum(self.likelihood.logpdf(self.data, x, self.data_sigma))

    def log_prior(self, x: np.ndarray) -> np.float64:
        return self.prior.logpdf(x)




class MetropolisHastings:
    def __init__(self, target_distribution: TargetDistribution, proposal_distribution: Proposal,
                 initialstate):
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
        self.uniform_rng = np.random.RandomState(seed=SEED)  # MT19937

    def acceptance_ratio(self, current: np.ndarray, proposed: np.ndarray) -> np.float64:
        """
        Acceptance ratio for the Metropolis-Hastings algorithm.
        :param current: Current state of the chain.
        :param proposed: Proposed state of the chain.
        :return: Acceptance ratio.
        """

        prior_ratio = self.target_distribution.log_prior(proposed) - self.target_distribution.log_prior(current)
        likelihood_ratio = self.target_distribution.log_likelihood(proposed) - self.target_distribution.log_likelihood(current)
        transition_ratio = (self.proposal_distribution.proposal_log_density(current, proposed) -
                            self.proposal_distribution.proposal_log_density(proposed, current))# Previous given new over new given previous

        # Print to see what we're getting
        print("Prior ratio:", prior_ratio)
        print("Likelihood ratio:", likelihood_ratio)
        print("Transition ratio:", transition_ratio)

        log_ratio = prior_ratio + likelihood_ratio + transition_ratio
        return min(np.float64(0), log_ratio)

    def __call__(self, n: int):
        """
        Run the Metropolis-Hastings algorithm for n iterations.
        :param n:
        :return:
        """
        # Resize if needed
        if self._index + n > self.max_size:
            new_max = max(self.max_size * 2, self._index + n)
            new_chain = np.empty((new_max, self.chain.shape[1]))
            new_chain[:self._index] = self.chain[:self._index]
            self.chain = new_chain
            self.max_size = new_max

        for i in range(n):
            current = self.chain[self._index - 1]
            proposed = self.proposal_distribution.propose(current)
            if np.log(self.uniform_rng.uniform()) < self.acceptance_ratio(current, proposed):
                self.chain[self._index] = proposed
            else:
                self.chain[self._index] = current
            self._index += 1


import matplotlib.pyplot as plt

# True parameter values for damped oscillator
true_params = np.array([0.2, 2.0])  # [damping, frequency]


# Generate synthetic data
def oscillator(t, params):
    """Damped oscillator solution"""
    damping, freq = params
    return np.exp(-damping * t) * np.cos(freq * t)


# Generate data points
t_obs = np.linspace(0, 10, 20)
true_signal = oscillator(t_obs, true_params)
noise_level = 0.1
noisy_data = true_signal + np.random.normal(0, noise_level, size=len(t_obs))


# Set up target distribution
class OscillatorTarget(TargetDistribution):
    def __init__(self, data, t, noise_sigma):
        self.data = data
        self.t = t
        self.noise_sigma = noise_sigma

        # Use uniform priors
        self.prior = sp.stats.uniform(loc=[0, 0], scale=[1, 5])
        # Gaussian likelihood
        self.likelihood = sp.stats.norm

    def log_likelihood(self, params):
        predicted = oscillator(self.t, params)
        return np.sum(self.likelihood.logpdf(self.data, predicted, self.noise_sigma))

    def log_prior(self, params):
        return np.sum(self.prior.logpdf(params))  # Sum the log probabilities


# Set up MCMC
target = OscillatorTarget(noisy_data, t_obs, noise_level)
proposal = Proposal(sp.stats.multivariate_normal,
                    scale=np.array([[0.01, 0],
                                    [0, 0.01]]))  # Small step size

# Initial guess
initial_state = np.array([0.1, 1.5])

# Run MCMC
mcmc = MetropolisHastings(target, proposal, initial_state)
mcmc(5000)  # Run for 5000 iterations

# Plot results
plt.figure(figsize=(12, 4))

# Plot data and fit
plt.subplot(121)
plt.plot(t_obs, noisy_data, 'k.', label='Data')
plt.plot(t_obs, true_signal, 'g-', label='True')
final_params = mcmc.chain[4000]  # Use a late sample
plt.plot(t_obs, oscillator(t_obs, final_params), 'r--', label='Estimated')
plt.legend()
plt.title('Data and Fit')

# Plot parameter traces
plt.subplot(122)
plt.plot(mcmc.chain[:4000, 0], mcmc.chain[:4000, 1], 'k.', alpha=0.1)
plt.plot(true_params[0], true_params[1], 'r*', markersize=10)
plt.xlabel('Damping')
plt.ylabel('Frequency')
plt.title('Parameter Space')

plt.show()