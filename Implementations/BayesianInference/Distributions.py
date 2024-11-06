import scipy as sp
import numpy as np
from typing import Tuple, Union
from .PRNG import RNG, SEED


class Proposal:
    def __init__(
        self, proposal_distribution: sp.stats.rv_continuous, scale: np.ndarray
    ):
        self.proposal_distribution = proposal_distribution
        self.proposal = RNG(SEED, proposal_distribution)
        if np.isscalar(scale):
            self.beta = np.sqrt(scale)
        else:
            self.beta = sp.stats.Covariance.from_cholesky(scale)  # L*x ~ N(0, Sigma)

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
        return self.prior.logpdf(x)
