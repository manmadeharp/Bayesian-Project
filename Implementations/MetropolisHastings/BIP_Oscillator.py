import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from Implementations.BayesianInference import SEED
from Implementations.BayesianInference import TargetDistribution, Proposal
from Implementations.BayesianInference.MetropolisHastings import MetropolisHastings

# The static SEED variable for reproducibility
Seed = 1

# True parameter values for damped oscillator
true_params = np.array([0.2, 2.0])  # [damping, frequency]


# Generate synthetic data
def oscillator(t, params):
    """Damped oscillator solution"""
    damping, freq = params
    return np.exp(-damping * t) * np.cos(freq * t)


# Generate data points
t_obs = np.linspace(0, 10, 100)
true_signal = oscillator(t_obs, true_params)
noise_level = 0.1
noise_rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(SEED)))

noisy_data = true_signal + noise_rng.normal(0, noise_level, size=len(t_obs))


class OscillatorTarget(TargetDistribution):
    def __init__(self, data, t, noise_sigma):
        # Use parent's structure as is
        prior = sp.stats.uniform(loc=[0, 0], scale=[1, 5])
        likelihood = sp.stats.norm
        super().__init__(prior, likelihood, data, noise_sigma)
        self.t = t

    def forward_model(self, params: np.ndarray) -> np.ndarray:
        """Transform parameters to predictions"""
        return oscillator(self.t, params)

    # Parent log_likelihood and log_prior stay the same!
    # Just use forward_model when needed:
    def log_likelihood(self, x: np.ndarray) -> np.float64:
        predicted = self.forward_model(x)
        return super().log_likelihood(predicted)

    def log_prior(self, x):
        return np.sum(super().log_prior(x))  # Sum the log probabilities


# Set up MCMC
target = OscillatorTarget(noisy_data, t_obs, noise_level)
proposal = Proposal(
    sp.stats.multivariate_normal, scale=np.array([[0.1, 0], [0, 0.1]])
)  # Small step size

# Initial guess
initial_state = np.array([0.1, 1.5])

# Run MCMC
mcmc = MetropolisHastings(target, proposal, initial_state)
mcmc(5000)

# Plot results
plt.figure(figsize=(12, 4))

# Plot data and fit
plt.subplot(121)
plt.plot(t_obs, noisy_data, "k.", label="Data")
plt.plot(t_obs, true_signal, "g-", label="True")
final_params = mcmc.chain[4000]  # Use a late sample
plt.plot(t_obs, oscillator(t_obs, final_params), "r--", label="Estimated")
plt.legend()
plt.title("Data and Fit")

# Plot parameter traces
plt.subplot(122)
plt.plot(mcmc.chain[:4000, 0], mcmc.chain[:4000, 1], "k.", alpha=0.1)
plt.plot(true_params[0], true_params[1], "r*", markersize=10)
plt.xlabel("Damping")
plt.ylabel("Frequency")
plt.title("Parameter Space")

plt.show()
