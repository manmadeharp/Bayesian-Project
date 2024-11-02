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
    def __init__(self, value: NDArray[np.float64], prior: Callable, scale):
        self.initial_value = value
        self.prior = prior
        


class RandomWalkMetropolisHastings:
    def __init__(self, prior, scale):


        # For proposals
        self.beta = scale

        if initial_value == None:
            self.chain = np.zeros((0, params.size))
        else:
            self.chain = np.atleast_2d(initial_value) # 2d for [ [initial_theta] ]

    def prior():

        return

    def likelihood():
        return
