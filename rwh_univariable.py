import inspect

# from rwh import LinearData
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Tuple, Callable

from panel.pane.vtk.synchronizable_serializer import linspace
from scipy import stats

def linear_model(x, m, c):
    return m * x + c

class LinearData:
    def __init__(self, m: float, b: float, sigma: float, size: int = 1):
        self.m = m
        self.b = b
        self.sigma = sigma
        self.size = size
        self.x = np.random.normal(0, 100, self.size) #np.linspace(0, 100, size)
        self.y = self.generate_data()

    def generate_data(self) -> NDArray[np.float64]:
        noise = np.random.normal(0, self.sigma, self.size)
        print(noise)
        return linear_model(self.x, self.m, self.b) + noise

    def plot_data(self):
        plt.scatter(self.x, self.y, alpha=0.5)
        plt.plot(np.linspace(0, 100, 100), linear_model(np.linspace(0, 100, 100), self.m, self.b), 'r-', label='True Line')
        plt.legend()
        plt.show()

    def true_cdf(self, x):
        return stats.norm.cdf(x, loc=self.m, scale=self.sigma)

# Poor Mixing
class RWMH:
    def __init__(self, prop_model: Callable, theta_0: NDArray[np.float64], sigma_0,  m: float, b: float, y: NDArray[np.float64]):
        self.model = prop_model
        self.theta: NDArray[np.float64] = np.array([theta_0])  # theta in a row array
        self.sigma = sigma_0
        self.samples = y.size
        self.m = m
        self.b = b
        self.data = y
        self.accept_count = 0
        self.accept_rate = np.array([0])

    def log_likelihood(self, theta):
        # Takes the Gaussian likelihood of the data and the model given the current theta
        return np.sum(-1 / 2 * ((self.data - self.model(theta, self.m, self.b)) ** 2 / self.sigma))

    def proposal_ratio(self, theta_proposed, theta_current):
        """
        Takes the ratio of the likelihood of the proposed theta to the current theta.
        The greater the ratio, the more likely the proposed theta is to be accepted.
        This is because the proposed theta is
        :param theta_proposed:
        :param theta_current:
        :return:
        """
        return self.log_likelihood(theta_proposed) - self.log_likelihood(theta_current)

    def acceptance_rate(self):
        # Acceptance rate
        rate = self.accept_count/self.theta.shape[0]
        # self.sigma = np.exp(1 * (rate - 0.234))
        self.accept_rate = np.append(self.accept_rate, rate)

    def sample(self):
        """
        Sample from the posterior distribution of the model parameters using the Random Walk Metropolis-Hastings algorithm.
        :return:
        """
        theta_proposed = np.random.normal(self.theta[-1], self.sigma, self.theta[-1].shape)
        alpha = self.proposal_ratio(theta_proposed, self.theta[-1]) # minimum is 0 instead of 1 since log(1) = 0
        if np.log(np.random.random()) < alpha:
            self.theta = np.vstack([self.theta, theta_proposed]) # add the proposed theta to the stack
            self.accept_count += 1
        else:
            self.theta = np.vstack([self.theta, self.theta[-1]])
        return self.theta

    def burning(self, number: int):
        """
        Remove the first 'burnin' number of samples from theta and adjust accept_count.

        :param number: number of samples to remove
        """
        if number >= self.theta.shape[0]:
            raise ValueError("Burning period cannot be longer than or equal to the total number of samples.")

        self.theta = self.theta[number:]
        self.accept_count = np.sum(self.theta[1:] != self.theta[:-1])
        self.accept_rate = self.accept_rate[number:]



    def empirical_distribution(self, x):
        """
        The empirical distribution function of the parameter samples.
        Calculated using the Gilenko-Cantelli Theorem since the params are i.i.d.
        :return:
        """
        return np.sum(np.sort(self.theta) < x) / self.theta.size

    def kolmogorov_smirnov(self, CDF: Callable):
        """
        The Kolmogorov-Smirnov test for the empirical distribution of the param.
        :return:
        """


    def plot_samples(self):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(np.linspace(1, self.theta.size, self.theta.size), self.theta[:], label='param')
        plt.plot(l.x, 'ro', label='true param value')
        plt.plot(l.y, 'bo', label='observed value')
        plt.legend()
        plt.title('Parameter Sample Path')

        plt.subplot(2, 2, 2)
        plt.hist(self.theta, bins=50)
        plt.title('Probability density')

        x_range = np.linspace(np.min(self.theta), np.max(self.theta), self.theta.size)

        plt.subplot(2, 2, 3)
        y = [self.empirical_distribution(I) for I in x_range]
        plt.plot(linspace(np.min(self.theta), np.max(self.theta), self.theta.size), y, label='Empirical CDF')
        plt.plot(l.x, self.empirical_distribution(l.x), 'bo', label='F(True Param Value)')
        plt.title('Empirical Distribution')
        plt.xlabel('θ')
        plt.ylabel('F(θ)')

        plt.subplot(2, 2, 4)

        kde = stats.gaussian_kde(np.sort(self.theta.flatten()))
        plt.plot(x_range, kde(x_range), label='Sample PDF (KDE)')

        true_pdf = stats.norm.pdf(x_range, loc=l.x, scale=l.sigma)
        plt.plot(x_range, true_pdf, label='True PDF', linestyle='--')

        plt.title('Sample PDF vs True PDF')
        plt.xlabel('θ')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

l = LinearData(2, 3, 10)
l.plot_data()
r = RWMH(linear_model, np.array([40]), 10000, l.m, l.b, l.y)

for i in range(100000):
    r.sample()
    r.acceptance_rate()

r.burning(1)
r.plot_samples()

def error_plots(x, x_hat):
    # Calculate absolute error for each iteration
    abs_error = np.abs(x_hat - x)

    # Calculate running mean of estimates
    running_mean = np.cumsum(x_hat) / np.arange(1, len(x_hat) + 1)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(abs_error, label='Absolute Error')
    plt.plot(running_mean, 'r--', label='Running Mean', linewidth=2)
    plt.axhline(np.mean(x_hat), color='g', linestyle=':', label='Mean Value')
    plt.axhline(l.x, color='r', linestyle=':', label='True Value')

    plt.legend()
    plt.title('Parameter Estimation Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error / Estimate')
    # plt.yscale('log')  # Use log scale for better visualization
    plt.show()

def acceptance_plot(acceptance):
    plt.plot(np.linspace(1, acceptance.size, acceptance.size), acceptance, label='value')
    plt.legend()
    plt.title('acceptance rate')
    plt.show()

error_plots(l.y, r.theta)
acceptance_plot(r.accept_rate)
#
#
# class RWMH:
#     def __init__(self, prop_model, theta_0, sigma_0, m, b, y, data_sigma):
#         self.model = prop_model
#         self.theta = np.array([theta_0]).reshape(-1, 1)
#         self.sigma = sigma_0
#         self.m = m
#         self.b = b
#         self.data = y.reshape(-1, 1)
#         self.data_sigma = data_sigma
#         self.accept_count = 0
#         self.accept_rate = []
#         self.log_likelihoods = []
#
#     def log_likelihood(self, theta):
#         predicted = self.model(self.m, theta, self.b)
#         ll = np.sum(-0.5 * ((self.data - predicted) / self.data_sigma) ** 2)
#         return ll
#
#     def proposal_ratio(self, theta_proposed, theta_current):
#         ll_proposed = self.log_likelihood(theta_proposed)
#         ll_current = self.log_likelihood(theta_current)
#         self.log_likelihoods.append(ll_current)
#         return ll_proposed - ll_current
#
#     def sample(self):
#         current_theta = self.theta[-1]
#         # Use a mixture of local and global proposals
#         if np.random.random() < 0.9:  # 90% local proposals
#             theta_proposed = current_theta + np.random.normal(0, self.sigma, current_theta.shape)
#         else:  # 10% global proposals
#             theta_proposed = np.random.uniform(0, 100, current_theta.shape)
#
#         alpha = min(self.proposal_ratio(theta_proposed, current_theta), 1)
#         if np.log(np.random.random()) < alpha:
#             self.theta = np.vstack([self.theta, theta_proposed])
#             self.accept_count += 1
#         else:
#             self.theta = np.vstack([self.theta, current_theta])
#
#         if self.theta.shape[0] % 1000 == 0:
#             print(f"Iteration {self.theta.shape[0]}, Current theta: {current_theta.flatten()}, Proposed: {theta_proposed.flatten()}, Accepted: {self.theta[-1].flatten() != current_theta.flatten()}")
#
#     def run_chain(self, n_iterations):
#         for _ in range(n_iterations):
#             self.sample()
#             if _ % 100 == 0:
#                 self.acceptance_rate()
#
#         # Check if the chain has moved at all
#         if np.all(self.theta[0] == self.theta[-1]):
#             print("Warning: The chain hasn't moved from its starting position!")
#
#     def acceptance_rate(self):
#         rate = self.accept_count / self.theta.shape[0]
#         self.accept_rate.append(rate)
#
#     def plot_diagnostics(self):
#         plt.figure(figsize=(15, 10))
#
#         plt.subplot(2, 2, 1)
#         for i in range(self.theta.shape[1]):
#             plt.plot(self.theta[:, i], label=f'x_{i}')
#         plt.title('Parameter Traces')
#         plt.legend()
#
#         plt.subplot(2, 2, 2)
#         plt.plot(self.accept_rate)
#         plt.title('Acceptance Rate')
#
#         plt.subplot(2, 2, 3)
#         plt.plot(self.log_likelihoods)
#         plt.title('Log-Likelihood Trace')
#
#         plt.subplot(2, 2, 4)
#         plt.hist(self.theta.flatten(), bins=50)
#         plt.title('Parameter Histogram')
#
#         plt.tight_layout()
#         plt.show()
#
#
# # Usage
# l = LinearData(2, 3, 10, size=5)
# l.plot_data()
#
# initial_x = np.random.uniform(0, 100, l.y.shape)
# r = RWMH(linear_model, initial_x, 1, l.m, l.b, l.y, l.sigma)
#
# r.run_chain(50000)
# r.plot_diagnostics()
#
# error_plots(l.x, r.theta)
