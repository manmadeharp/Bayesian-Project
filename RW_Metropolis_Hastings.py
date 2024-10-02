import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable
import matplotlib.pyplot as plt


class LinearDataModel:
    def __init__(self, true_slope: float, true_intercept: float, noise_std: float, size: int = 100):
        self.true_slope = true_slope
        self.true_intercept = true_intercept
        self.noise_std = noise_std
        self.size = size
        self.x = np.linspace(0, 10, size)
        self.y = self.generate_data()

    def generate_data(self) -> NDArray[np.float64]:
        return self.true_slope * self.x + self.true_intercept + np.random.normal(0, self.noise_std, self.size)

    def plot_data(self):
        plt.scatter(self.x, self.y, alpha=0.5)
        plt.plot(self.x, self.true_slope * self.x + self.true_intercept, 'r-', label='True Line')
        plt.legend()
        plt.show()


def log_likelihood(theta: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64], noise_std: float) -> float:
    slope, intercept = theta
    y_pred = slope * x + intercept
    return -0.5 * np.sum(((y - y_pred) / noise_std) ** 2)


def log_prior(theta: NDArray[np.float64]) -> float:
    slope, intercept = theta
    if -10 < slope < 10 and -10 < intercept < 10:
        return 0.0
    return -np.inf


def log_posterior(theta: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64], noise_std: float) -> float:
    return log_likelihood(theta, x, y, noise_std) + log_prior(theta)


def propose(theta: NDArray[np.float64], proposal_std: float) -> NDArray[np.float64]:
    return theta + np.random.normal(0, proposal_std, size=theta.shape)


def rwmh_sampler(
        log_post: Callable,
        initial_theta: NDArray[np.float64],
        n_samples: int,
        proposal_std: float,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        noise_std: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    theta = initial_theta
    samples = np.zeros((n_samples, len(initial_theta)))
    log_post_values = np.zeros(n_samples)

    for i in range(n_samples):
        theta_proposal = propose(theta, proposal_std)
        log_post_current = log_post(theta, x, y, noise_std)
        log_post_proposal = log_post(theta_proposal, x, y, noise_std)

        if np.log(np.random.random()) < log_post_proposal - log_post_current:
            theta = theta_proposal

        samples[i] = theta
        log_post_values[i] = log_post_current

    return samples, log_post_values


# Example usage
model = LinearDataModel(true_slope=2, true_intercept=12, noise_std=0.5, size=100)
model.plot_data()

initial_theta = np.array([3.0, 10.0])
n_samples = 1011
proposal_std = 0.1

samples, log_post_values = rwmh_sampler(
    log_posterior, initial_theta, n_samples, proposal_std,
    model.x, model.y, 1
)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(samples[:, 0], label='Slope')
plt.plot(samples[:, 1], label='Intercept')
plt.legend()
plt.title('Parameter Traces')

plt.subplot(122)
plt.hist2d(samples[:, 0], samples[:, 1], bins=50, cmap='Blues')
plt.colorbar()
plt.xlabel('Slope')
plt.ylabel('Intercept')
plt.title('Joint Posterior')

plt.tight_layout()
plt.show()

print(f"True parameters: slope={model.true_slope}, intercept={model.true_intercept}")
print(f"Estimated parameters: slope={np.mean(samples[:, 0]):.2f}, intercept={np.mean(samples[:, 1]):.2f}")

