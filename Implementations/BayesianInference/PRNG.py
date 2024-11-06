from numpy.random import SeedSequence, Generator, Philox
from scipy.stats import rv_continuous
from typing import Optional

# Static Values
SEED = 1000


# Parent RNG class
class RNG:
    def __init__(self, seed: Optional[int], distribution: rv_continuous):
        self.ss = SeedSequence(seed)
        self.rg = Generator(Philox(self.ss))  # Use Philox for parallel applications
        self.rng_distribution = distribution

    def __call__(self, loc, scale, size: Optional[int] = None, *args, **kwargs):
        return self.rng_distribution.rvs(
            loc, scale, size=size, random_state=self.rg, *args, **kwargs
        )
