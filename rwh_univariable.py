import inspect
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

from scipy import stats

def linear_model(x, m, c):
    return m * x + c

"""
Important changes, adding the prior to the calculations smooths out the distribution and improves the shape of the pdf
The greater the variance the more the state space is explored and the lower the acceptance rate is.
The further the current parameter value is from the true value the greater the rate of acceptance.
The acceptance rate is a measure of the efficiency of the algorithm.
"""

class LinearData:
    def __init__(self, m: float, b: float, sigma: float, size: int = 1):
        self.m = m
        self.b = b
        self.sigma = sigma
        self.size = size
        self.x = np.random.uniform(0, 100, self.size) #np.linspace(0, 100, size)
        self.y = self.generate_data()

    def generate_data(self) -> NDArray[np.float64]:
        noise = np.random.normal(0, self.sigma, self.size)
        return linear_model(self.x, self.m, self.b) + noise

    def plot_data(self):
        plt.scatter(self.x, self.y, alpha=0.5)
        plt.scatter((np.mean(l.y) - l.b)/l.m, self.y, label='Prior: Observation on Model')
        plt.plot(np.linspace(0, 100, 100), linear_model(np.linspace(0, 100, 100), self.m, self.b), 'r-', label='True Line')
        plt.legend()
        plt.show()

    def true_cdf(self, x):
        return stats.norm.cdf(x, loc=self.m, scale=self.sigma)

class RWMH:
    def __init__(self, prop_model: Callable, theta_0: NDArray[np.float64],
                 sigma_0, m: float, b: float, y: NDArray[np.float64],
                 prior_theta: float, scaling: float):
        self.model = prop_model
        self.theta: NDArray[np.float64] = np.array([theta_0])  # theta in a row array
        self.sigma = sigma_0
        self.samples = y.size
        self.m = m
        self.b = b
        self.data = y
        self.accept_count = 0
        self.accept_rate = np.array([0])
        self.prior_theta = prior_theta
        self.beta = scaling

    def log_likelihood(self, theta):
        # Takes the Gaussian likelihood of the data and the model given the current theta
        return np.sum(-1 / 2 * ((self.data - self.model(theta, self.m, self.b)) / self.sigma) ** 2) # Need to add self.beta`

    def proposal_ratio(self, theta_proposed, theta_current):
        """
        Takes the ratio of the likelihood of the proposed theta to the current theta.
        The greater the ratio, the more likely the proposed theta is to be accepted.
        This is because the proposed theta is
        :param theta_proposed:
        :param theta_current:
        :return:
        """
        lower = self.prior_theta - 20
        log_prior = stats.uniform.logpdf(theta_proposed, loc=lower, scale=40) - stats.uniform.logpdf(theta_current, loc=lower, scale=40)

        return self.log_likelihood(theta_proposed) - self.log_likelihood(theta_current) + log_prior

    def acceptance_rate(self):
        # Acceptance rate
        rate = self.accept_count/self.theta.size
        self.accept_rate = np.append(self.accept_rate, rate)

    def scale_beta(self):
        self.beta *= np.exp( 0.02*(self.accept_rate[-1] - 0.234) )

    def sample(self):
        """
        Sample from the posterior distribution of the model parameters using the Random Walk Metropolis-Hastings algorithm.
        :return:
        """
        #theta_proposed = np.random.normal(self.theta[-1], self.beta, self.theta[-1].shape)
        theta_proposed = np.random.uniform(0, 100)
        alpha = self.proposal_ratio(theta_proposed, self.theta[-1]) # minimum is 0 instead of 1 since log(1) = 0
        uniform = np.log(np.random.random())
        if uniform < alpha:
            self.theta = np.vstack([self.theta, theta_proposed]) # add the proposed theta to the stack
            self.accept_count += 1
        else:
            self.theta = np.vstack([self.theta, self.theta[-1]])
        return self.theta

    def burning(self, number: int):
        """
        Remove the first number of samples from theta and adjust accept_count.

        :param number: number of samples to remove
        """
        if number >= self.theta.shape[0]:
            raise ValueError("Burning period cannot be longer than or equal to the total number of samples.")
        print("Burning period: ", number)
        print("Theta Before: ", self.theta)
        self.theta = self.theta[number:]
        print("Theta After: ", self.theta)
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


    def autocorrelation(self, lag: int = 50) -> NDArray[np.float64]:
        """
        Calculate autocorrelation.
        """
        y = self.theta.flatten() - np.mean(self.theta)
        result = np.correlate(y, y, mode='full')
        result = result[result.size // 2:]
        return result[:lag] / result[0]

    def plot_samples(self):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(np.linspace(1, self.theta.size, self.theta.size), self.theta[:], label='param')
        plt.plot(l.x, 'ro', label='true param value')
        plt.legend()
        plt.title('Parameter Sample Path')

        plt.subplot(2, 2, 2)
        plt.hist(self.theta, bins=50)
        plt.title('Parameter Sample Histogram')
        x_range = np.linspace(np.min(self.theta), np.max(self.theta), self.theta.size)

        plt.subplot(2, 2, 3)
        y = [self.empirical_distribution(I) for I in x_range]
        plt.plot(np.linspace(np.min(self.theta), np.max(self.theta), self.theta.size), y, label='Empirical CDF')
        plt.plot(l.x, self.empirical_distribution(l.x), 'bo', label='F(True Param Value)')
        plt.title('Empirical Cumulative Distribution')
        plt.xlabel('θ')
        plt.ylabel('F(θ)')
        plt.legend()
        plt.subplot(2, 2, 4)

       # kde_quarter = stats.gaussian_kde(self.theta[:self.theta.size//4].flatten())
       # plt.plot(x_range, kde_quarter(x_range), label='Sample PDF (KDE) quarter', linestyle='--')

       # kde_half = stats.gaussian_kde(self.theta[:self.theta.size//2].flatten())
       # plt.plot(x_range, kde_half(x_range), label='Sample PDF (KDE) half', linestyle='--')

       # kde_qu_th = stats.gaussian_kde(self.theta[:3*self.theta.size//4].flatten(), bw_method='scott')
       # plt.plot(x_range, kde_qu_th(x_range), label='Sample PDF (KDE) third quarter', linestyle='--')
        
       # print("start kde")
       # kde_final = stats.gaussian_kde(self.theta.flatten())
       # plt.plot(x_range, kde_final(x_range), label='Sample PDF (KDE)', linestyle='-', color='b')

        true_pdf = stats.norm.pdf(x_range, loc=l.x, scale=l.sigma)
        plt.plot(x_range, true_pdf, label='True PDF', linestyle='-', color='r')
        #print("finish kde")

        plt.title('Sample PDF vs True PDF')
        plt.xlabel('θ')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    def plot_autocorrelation(self, lag: int = 50):
        """
        Plot the autocorrelation function.
        """
        acf = self.autocorrelation(lag)
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(acf)), acf, alpha=0.5)
        plt.plot(range(len(acf)), acf, 'r--', linewidth=2)
        plt.title('Autocorrelation Function')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.show()

l = LinearData(2, 3, 5)

l.plot_data()

r = RWMH(linear_model, np.array([0]), l.sigma, l.m, l.b, l.y, l.x, (np.mean(l.y) - l.b)/l.m ) # was l.sigma*10 but changed due to Efficient Metropolis Jumping Rules


beta_values = []
for i in range(45900):
    #    if i == 5000:
    #            r.beta = 10
    #    if i == 10000:
    #            r.beta = 7
    #    if i == 15000:
    #            r.beta = 5

    r.sample()
    r.acceptance_rate()
    r.scale_beta()
    beta_values.append(r.beta) #= np.vstack([beta_values, r.beta])

r.burning(100)

r.plot_samples()

def error_plots(x, x_hat):
    # Calculate absolute error for each iteration
    abs_error = np.abs(x_hat - x)
    # Calculate running mean of estimates
    running_mean = np.cumsum(x_hat) / np.arange(1, len(x_hat) + 1)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(abs_error, label='Absolute Error')
    plt.title('Parameter Estimation Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error / Estimate')
    plt.subplot(1, 2, 2)
    plt.plot(running_mean, 'r--', label='Running Mean', linewidth=2)
    plt.axhline(np.mean(x_hat), color='g', linestyle=':', label='Mean Value')
    plt.axhline(l.x, color='r', linestyle=':', label='True Value')
    plt.title('Mean Values')
    plt.xlabel('Iteration')
    plt.ylabel('$\\theta$')
    plt.legend()

    # plt.yscale('log')  # Use log scale for better visualization
    plt.show()

def acceptance_plot(acceptance):
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(1, acceptance.size, acceptance.size), acceptance, label='value')
    plt.axhline(0.234, color='r', linestyle='-', label='Optimal Acceptance')
    plt.legend()
    plt.title('acceptance rate')
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(beta_values)), beta_values, label='Scaling Values')
    plt.axhline(np.mean(beta_values), color='r', linestyle='-', label='Optimal Scaling for 0.234')
    plt.legend()
    plt.title("Beta Values")
    plt.show()

error_plots(l.x, r.theta)
acceptance_plot(r.accept_rate)
r.plot_autocorrelation()



# def particle_trajectory(particle: NDArray[np.float64]):



# r2 = RWMH(linear_model, np.array(np.mean(r.theta)), 2.4, l.m, l.b, l.y) # was l.sigma*10 but changed due to Efficient Metropolis Jumping Rules
#
# for i in range(8000):
#     # if i == 10005:
#     #     r.sigma = l.sigma*2
#     # if i == 20000:
#     #     r.sigma = l.sigma
#     r2.sample()
#     r2.acceptance_rate()
#
# r2.burning(60)
#
# r2.plot_samples()
#
# error_plots(l.x, r2.theta)
# acceptance_plot(r2.accept_rate)
# r2.plot_autocorrelation()
