import sys
import os

import numpy as np
import scipy.stats as stats
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MetropolisHastings')))
from MetropolisHastings import RNG, Proposal, TargetDistribution, MetropolisHastings  # Import your classes here


class TestRNG(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.n = 5
        self.distribution = stats.norm
        self.rng = RNG(self.seed, self.n, self.distribution)

    def test_rng_generation(self):
        loc = 0
        scale = 1
        samples = self.rng(loc=loc, scale=scale)
        self.assertEqual(samples.shape, (self.n,))
        self.assertTrue(np.all(np.isfinite(samples)))


class TestProposal(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.parameter = np.array([0.0])
        self.proposal_distribution = stats.norm
        self.proposal = Proposal(self.proposal_distribution, self.parameter)

    def test_propose(self):
        current = np.array([1.0])
        proposal_samples = self.proposal.propose(current)
        self.assertEqual(proposal_samples.shape, (self.proposal.proposal_dimension,))
        self.assertTrue(np.all(np.isfinite(proposal_samples)))


class TestTargetDistribution(unittest.TestCase):
    def setUp(self):
        self.prior = stats.norm(loc=0, scale=1)
        self.likelihood = stats.norm(loc=1, scale=1)
        self.data = 1.5
        self.sigma = 0.5
        self.target_dist = TargetDistribution(self.prior, self.likelihood, self.data, self.sigma)

    def test_log_likelihood(self):
        x = np.array([1.0])
        log_likelihood = self.target_dist.log_likelihood(x)
        self.assertTrue(np.isfinite(log_likelihood))

    def test_log_prob(self):
        x = np.array([1.0])
        log_prob = self.target_dist.log_prob(x)
        self.assertTrue(np.isfinite(log_prob))


class TestMetropolisHastings(unittest.TestCase):
    def setUp(self):
        self.target_dist = TargetDistribution(
            prior=stats.norm(loc=0, scale=1),
            likelihood=stats.norm(loc=1, scale=1),
            data=1.5,
            sigma=0.5
        )
        self.proposal_dist = stats.norm(loc=0, scale=1)
        self.initial_state = np.array([0.0])
        self.mh = MetropolisHastings(self.target_dist, self.proposal_dist, self.initial_state)

    def test_initialization(self):
        self.assertTrue(np.array_equal(self.mh.initial_state, self.initial_state))
        self.assertEqual(self.mh.chain.shape, (1, 1))  # Initial state shape should be (1, dimension)

    def test_prior(self):
        # Implement a simple test for the prior method
        # Note: This method currently has no implementation; this is just a placeholder
        pass


if __name__ == "__main__":
    unittest.main()
