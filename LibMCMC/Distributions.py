from typing import Tuple, Union

import numpy as np
import scipy as sp

from PRNG import RNG, SEED

## TODO: I need to Add Adaptation MCMC Diagnostics As well.
#    # Plot 5: Adaptation factor
#    adapt_factors = []
#   for i in range(len(chain)):
#       sampler._index = i
#       adapt_factors.append(sampler.get_adaptation_weight())
#   axes[1, 1].plot(adapt_factors)
#   axes[1, 1].set_xlabel("Iteration")
#   axes[1, 1].set_ylabel("Adaptation Weight")
#   axes[1, 1].set_title("Adaptation Weight Decay")


class Proposal:
    def __init__(self, proposal_distribution: sp.stats.rv_continuous, scale):
        self.proposal_distribution = proposal_distribution
        self.proposal = RNG(SEED // 2, proposal_distribution)
        if np.isscalar(scale):
            print("scalar")
            self.beta = np.sqrt(scale)
        else:
            self.beta = scale

    #            print("Cholesky")
    #            self.beta = sp.stats.Covariance.from_cholesky(scale)  # L*x ~ N(0, Sigma)

    def propose(self, current: np.ndarray):
        return self.proposal(current, self.beta)

    def proposal_log_density(
        self,
        state: np.ndarray,
        loc: np.ndarray,
    ) -> np.float64:
        return self.proposal_distribution.logpdf(state, loc, self.beta)


# Test Proposal
# test = Proposal(sp.stats.multivariate_normal, np.array([[1, 2], [2, 1]]))
# print(test.propose(np.array([1.0, 12])))
# print(test.proposal_log_density(np.array([1.0, 12]), np.array([1.0, 12])))


class TargetDistribution:
    def __init__(
        self,
        prior: sp.stats.rv_continuous,
        likelihood: sp.stats.rv_continuous,
        data,
        sigma: float,
    ):
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
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.reshape(-1, 1)

        if not hasattr(self.prior, "mean"):  # Quick way to check if it's non-frozen
            return self.prior.logpdf(x[np.newaxis, :])

        return self.prior.logpdf(x)


class BayesInverseGammaVarianceDistribution(TargetDistribution):
    """Target distribution with inverse gamma prior on noise variance"""

    def __init__(
        self,
        prior: sp.stats.rv_continuous,
        likelihood: sp.stats.rv_continuous,
        data,
        alpha: float = 2.0,  # Shape parameter for inverse gamma
        beta: float = 1.0,  # Scale parameter for inverse gamma
    ):
        # Initialize parent without sigma since we're marginalizing it
        super().__init__(prior, likelihood, data, sigma=None)

        # Store inverse gamma hyperparameters
        self.alpha = alpha
        self.beta = beta

    def log_likelihood(self, x: np.ndarray) -> np.float64:
        """
        Compute marginalized log likelihood integrating out σ²
        p(y|θ) = ∫ p(y|θ,σ²)p(σ²)dσ²
        """
        residuals = self.data - x  # Or self.forward_model(x) for complex models
        n = len(residuals)
        RSS = np.sum(residuals**2)

        # Updated inverse gamma parameters
        alpha_post = self.alpha + n / 2
        beta_post = self.beta + RSS / 2

        # Log marginal likelihood (multivariate t-distribution)
        log_lik = -alpha_post * np.log(beta_post)
        log_lik += sp.special.gammaln(alpha_post) - sp.special.gammaln(self.alpha)
        log_lik -= (n / 2) * np.log(2 * np.pi)

        return np.float64(log_lik)

    def sample_variance_posterior(self, x: np.ndarray) -> float:
        """Sample from conditional posterior of σ² given parameters"""
        residuals = self.data - x
        n = len(residuals)
        RSS = np.sum(residuals**2)

        # Posterior inverse gamma parameters
        alpha_post = self.alpha + n / 2
        beta_post = self.beta + RSS / 2

        return sp.stats.invgamma.rvs(alpha_post, scale=beta_post)
