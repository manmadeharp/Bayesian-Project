import random
from typing import Callable, AnyStr
import inspect
import numpy as np
from numpy.typing import NDArray
import scipy as sp

def simple_model(a, b, c, d, e):
    f = a*(b - c) - d*e
    return f

class Forward_Model:
    def __init__(self, model: Callable):
        self.model = model
        self.param_names = list(inspect.signature(model).parameters.keys())

class Parameter:
    def __init__(self, value: NDArray[np.float64], prior: Callable):
        self.initial_value = value
        self.prior = prior


class RandomWalkMetropolisHastings:
    def __init__(self, params: NDArray[np.float64], proposal_distribution, initial_value: NDArray[np.float64] | None = None):
        self.params = params
        self.proposal_distribution = proposal_distribution

        if initial_value == None:
            self.chain = np.zeros((0, params.size))
        else:
            self.chain = np.atleast_2d(initial_value)

    def prior():
        return

    def likelihood():
        return
