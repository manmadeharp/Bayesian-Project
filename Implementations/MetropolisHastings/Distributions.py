import scipy as stats
import numpy as np
from typing import Tuple, Union

class ContinuousUnivariateDistribution():
    def __init__(self, pdf, cdf):
        self.pdf = pdf
        self.cdf = cdf

    def pdf(self, *args, **kwargs):
        return self.pdf(*args, **kwargs)

    def cdf(self, *args, **kwargs):
        return self.cdf(*args, **kwargs)


class ContinuousMultivariateDistribution():
    def __init__(self, pdf, cdf):
        self.pdf = pdf
        self.cdf = cdf

    def pdf(self, *args, **kwargs):
        return self.pdf(*args, **kwargs)

    def cdf(self, *args, **kwargs):
        return self.cdf(*args, **kwargs)


class GaussianDistribution():

    def __init__(self, mean: Union[float, np.ndarray], 
                 covariance: Union[float, np.ndarray]):
        """
        Initialize a Gaussian distribution.

        :param mean: For univariate, a float. For multivariate, a 1D numpy array.
        :param covariance: For univariate, a float (variance). 
                           For multivariate, a 2D numpy array (covariance matrix).
        """
        self.mean = np.atleast_1d(mean)
        self.covariance = np.atleast_2d(covariance)
        
        if self.mean.shape[0] != self.covariance.shape[0]:
            raise ValueError("Dimensions of mean and covariance must match.")
        
        self.dim = self.mean.shape[0]
        self.is_univariate = self.dim == 1

    def log_prob(self, x: Union[float, np.ndarray]) -> float:
        """
        Returns the .

        :param x: The inputs.
        :return: The value of the log pdf at the input.
        """
        x = np.atleast_1d(x)
        if x.shape[-1] != self.dim:
            raise ValueError(f"Input dimension ({x.shape[-1]}) must match distribution dimension ({self.dim}).")
        
        if self.is_univariate:
            return stats.norm.logpdf(x, loc=self.mean[0], scale=np.sqrt(self.covariance[0, 0]))
        else:
            return stats.multivariate_normal.logpdf(x, mean=self.mean, cov=self.covariance)

    def conditional_prob(self, x: Union[float, np.ndarray], 
                         y: Union[float, np.ndarray]) -> float:
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        if x.shape[-1] != self.dim:
            raise ValueError(f"Input dimension ({x.shape[-1]}) must match distribution dimension ({self.dim}).")

        if self.is_univariate:
            return stats.norm.logpdf(x, loc=y, scale=np.sqrt(self.covariance[0, 0]))
        else:
            return stats.multivariate_normal.logpdf(x, mean=y, cov=self.covariance)

    def sample(self, size: Union[int, Tuple[int, ...]] = 1) -> np.ndarray:
        """
        Generate random samples from the Gaussian distribution.

        :param size: The number of samples to generate. 
                     Can be an integer or a tuple for multiple dimensions.
        :return: Random samples from the distribution.
        """
        if self.is_univariate:
            return stats.norm.rvs(loc=self.mean[0], scale=np.sqrt(self.covariance[0, 0]), size=size)
        else:
            return stats.multivariate_normal.rvs(mean=self.mean, cov=self.covariance, size=size)




