import random
from typing import Callable, AnyStr
import inspect
import numpy as np
from numpy.typing import NDArray
import scipy as sp


def simple_model(a, b, c, d, e):
    f = a * (b - c) - d * e
    return f


class ForwardModel:
    """
    This class is used to wrap a model function to provide methods to a mathematical model that make its components explicitly callable and accessible.
    It also should eventually allow for the augmentation of more complex models.
    """

    def __init__(self, model: Callable):
        self.model = model
        self.sig = inspect.signature(model)
        self.param_names = list(inspect.signature(model).parameters.keys())

    def __call__(self, *args, **kwargs):
        bound_args = self.sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return self.model(*bound_args.args, **bound_args.kwargs)

    def return_params(self):
        return self.param_names


class DiscreteParameter(sp.stats.rv_discrete):
    """
        The Discrete Parameter class is a more abstract class that uses numpys rvs abstract classes to define it's , but instead provides us with a way to modularise the
        choices made about parameters, such as:
        - Whether it should be allowed to vary (degrees of freedom)
        - Whether it is scalar or vector
        - If we want to impose a prior on it
        """

    def __init__(self, value: NDArray[np.float64], prior: Callable, ):

        self.initial_value = value  # Scalar or vector
        self.distribution = prior  # This is an optional argument, if we want to impose a prior on the parameter it will become a degree of freedom.


class ContinuousParameter(sp.stats.rv_continuous):
    """
    The Parameter class is a more abstract class that uses numpys rvs abstract classes to define it's , but instead provides us with a way to modularise the
    choices made about parameters, such as:
    - Whether it should be allowed to vary (degrees of freedom)
    - Whether it is scalar or vector
    - If we want to impose a prior on it
    """

    def __init__(self, value: NDArray[np.float64], prior: Callable, ):
        self.initial_value = value  # Scalar or vector
        self.distribution = prior  # This is an optional argument, if we want to impose a prior on the parameter it will become a degree of freedom.


class RandomWalkMetropolisHastings:
    def __init__(self, prior, proposal_scale: np.float64, initial_value: NDArray[np.float64]):
        # For proposals
        self.beta = proposal_scale
        # For chain
        self.chain = np.atleast_2d(initial_value)  # 2d for [ [initial_theta] ]


    def prior(self):
        return

    def likelihood(self):
        return

    def proposal(self):
        return

    def __call__(self, *args, **kwargs):
        return

    # Random CHange


model = Forward_Model(simple_model)
print(model.return_params())
