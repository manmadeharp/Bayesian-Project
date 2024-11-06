import numpy as np
import scipy.stats as sp
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BayesianInference.PRNG import RNG, SEED
from BayesianInference.Distributions import TargetDistribution, Proposal
from BayesianInference.MetropolisHastings import MetropolisHastings


class ReactionDiffusionTarget(TargetDistribution):
    """Target distribution for the reaction-diffusion inverse problem"""

    def __init__(self, data, x_points, noise_sigma, boundary_conditions):
        # Set up the prior for [r1, r2, D] parameters
        # Use uniform priors over reasonable ranges for each parameter
        prior = sp.uniform(loc=[0, 0, 0], scale=[5, 5, 1])
        likelihood = sp.t  # Using t-distribution for robustness
        super().__init__(prior, likelihood, data, noise_sigma)

        self.x = x_points
        self.bc_a, self.bc_b = boundary_conditions

    def reaction_term(self, u, params):
        """Cubic reaction term R(u) = r1*u - r2*u^3"""
        r1, r2, _ = params
        return r1 * u - r2 * u ** 3

    def solve_steady_state(self, params):
        """Solve the steady state reaction-diffusion equation"""
        D = params[2]  # Diffusion coefficient

        def ode_system(x, y):
            # y[0] is u, y[1] is u'
            return np.vstack((y[1], -self.reaction_term(y[0], params) / D))

        def boundary_conditions(ya, yb):
            return np.array([ya[0] - self.bc_a, yb[0] - self.bc_b])

        # Initial guess for solution
        y = np.zeros((2, len(self.x)))
        y[0] = np.linspace(self.bc_a, self.bc_b, len(self.x))

        try:
            sol = solve_bvp(ode_system, boundary_conditions, self.x, y)
            return sol.sol(self.x)[0]
        except:
            return None

    def forward_model(self, params):
        """Forward model implementation required by parent class"""
        solution = self.solve_steady_state(params)
        if solution is None:
            # Return something that will make log_likelihood very negative
            return np.full_like(self.data, np.inf)
        return solution

    def log_likelihood(self, params):
        """Override parent log_likelihood to handle solver failures"""
        predicted = self.forward_model(params)
        if np.any(np.isinf(predicted)):
            return -np.inf
        return super().log_likelihood(predicted)


def generate_synthetic_data(x_points, true_params, noise_sigma):
    """Generate synthetic data for testing"""
    model = ReactionDiffusionTarget(None, x_points, noise_sigma, (0, 0))
    true_solution = model.solve_steady_state(true_params)

    # Use the configured PRNG from your library
    noise_rng = RNG(SEED, sp.norm)
    noisy_data = true_solution + noise_rng(0, noise_sigma, size=len(true_solution))
    return noisy_data, true_solution


def main():
    # Problem setup
    x_points = np.linspace(0, 1, 100)
    noise_sigma = 0.1
    true_params = np.array([2.0, 1.0, 0.1])  # [r1, r2, D]

    # Generate synthetic data
    noisy_data, true_solution = generate_synthetic_data(x_points, true_params, noise_sigma)

    # Set up target and proposal distributions
    target = ReactionDiffusionTarget(noisy_data, x_points, noise_sigma, (0, 0))

    # Scale matrix for proposal - important for good mixing
    scale = np.diag([0.1, 0.1, 0.01])  # Smaller steps for diffusion coefficient
    proposal = Proposal(sp.multivariate_normal, scale=scale)

    # Initial state - start reasonably close to true values
    initial_state = np.array([1.5, 0.8, 0.08])

    # Run MCMC
    mcmc = MetropolisHastings(target, proposal, initial_state)
    n_iterations = 10000
    mcmc(n_iterations)

    # Plot results
    plt.figure(figsize=(15, 5))

    # Parameter traces
    plt.subplot(121)
    labels = ['r₁', 'r₂', 'D']
    for i in range(3):
        plt.plot(mcmc.chain[:, i], label=labels[i])
    plt.legend()
    plt.title('Parameter Traces')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')

    # Data and fit comparison
    plt.subplot(122)
    plt.plot(x_points, noisy_data, 'k.', alpha=0.5, label='Data')
    plt.plot(x_points, true_solution, 'g-', label='True')

    # Use final parameters for fit
    final_params = mcmc.chain[-1]
    final_solution = target.solve_steady_state(final_params)
    plt.plot(x_points, final_solution, 'r--', label='MCMC Fit')

    plt.legend()
    plt.title('Data and Model Fit')
    plt.xlabel('x')
    plt.ylabel('u(x)')

    plt.tight_layout()
    plt.show()

    # Print results
    print("\nResults:")
    print("True parameters:", true_params)
    print("Estimated parameters (mean of last 1000 samples):")
    print(np.mean(mcmc.chain[-1000:], axis=0))
    print("\nParameter standard deviations:")
    print(np.std(mcmc.chain[-1000:], axis=0))


if __name__ == "__main__":
    main()