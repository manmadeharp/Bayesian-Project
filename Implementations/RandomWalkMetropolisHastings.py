import random
import numpy as np
from numpy.typing import NDArray
import scipy as sp


class RandomWalkMetropolisHastings:
    def __init__(self, params: NDArray[np.float64]):
        self.params = params
        self.chain = np.zeros((0, self.params.size))
