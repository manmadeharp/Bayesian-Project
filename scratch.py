import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from scipy import stats
import inspect

class DataGenerator:
    def __init__(self, , model: Callable, **kwargs):
        self.model = model
        self.params = kwargs

    def __call__(self):
        sig = inspect.signature(self.model)
        model_params = {k: self.params[k] for k, v in sig.parameters.items() if k in self.params}
        return self.model(**model_params)

print(DataGenerator(linear_model, x=1, m=2, c=3)())