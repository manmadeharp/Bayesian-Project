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
        prior = sp.uniform(loc=[0, 0, 0], scale=[5, 5, 1])
        likelihood = sp.t  # Using t-distribution with 6 degrees of freedom
        super().__init__(prior, likelihood, data, noise_sigma)

        self.x = x_points
        self.bc_a, self.bc_b = boundary_conditions

    def log_likelihood(self, x) -> np.float64:
        """Override parent log_likelihood to handle solver failures"""
        predicted = self.forward_model(x)
        if np.any(np.isinf(predicted)):
            return np.float64(-np.inf)
        # Use t-distribution with 6 degrees of freedom
        return np.float64(
            np.sum(
                self.likelihood.logpdf(
                    self.data - predicted,  # residuals
                    df=4,  # degrees of freedom
                    loc=0,  # mean
                    scale=self.data_sigma,  # scale
                )
            )
        )

    def log_prior(self, x: np.ndarray) -> np.float64:
        """Log prior probability"""
        if np.any(x <= 0):  # Parameters must be positive
            return np.float64(-np.inf)
        return np.float64(np.sum(super().log_prior(x)))

    def forward_model(self, params):
        solution = self.solve_steady_state(params)
        if solution is None:
            return np.full_like(self.data, np.inf)
        return solution

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
            if not sol.success:
                return None
            return sol.sol(self.x)[0]
        except:
            return None

    def reaction_term(self, u, params):
        """Cubic reaction term R(u) = r1*u - r2*u^3"""
        r1, r2, _ = params
        return r1 * u - r2 * u**3


def generate_synthetic_data(x_points, true_params, noise_sigma, boundary_conditions):
    """Generate synthetic data for testing"""
    model = ReactionDiffusionTarget(None, x_points, noise_sigma, boundary_conditions)
    true_solution = model.solve_steady_state(true_params)

    if true_solution is None:
        raise ValueError("Failed to solve ODE for true parameters")

    # Use the configured PRNG from your library
    noise_rng = RNG(SEED, sp.norm)
    noisy_data = true_solution + noise_rng(0, noise_sigma, len(true_solution))
    return noisy_data, true_solution


if __name__ == "__main__":
    # Problem setup
    x_points = np.linspace(0, 1, 100)
    noise_sigma = 0.2
    true_params = np.array([2.0, 1.0, 0.1])  # [r1, r2, D]
    boundary_conditions = (0.2, 0.2)  # Non-zero but moderate boundary conditions

    # Generate synthetic data
    noisy_data, true_solution = generate_synthetic_data(
        x_points, true_params, noise_sigma, boundary_conditions
    )
    print("True solution shape:", true_solution.shape)
    print("True solution range:", np.min(true_solution), np.max(true_solution))

    # Set up MCMC with same boundary conditions
    target = ReactionDiffusionTarget(
        noisy_data, x_points, noise_sigma, boundary_conditions
    )
    scale = np.diag([0.6, 0.6, 0.1])
    proposal = Proposal(sp.multivariate_normal, scale=scale)
    initial_state = np.array([1.5, 0.8, 0.08])
    mcmc = MetropolisHastings(target, proposal, initial_state)

    # Run MCMC
    n_iterations = 50000
    mcmc(n_iterations)

    # Debug chain shape
    print("Chain shape:", mcmc.chain.shape)
    print("Active chain shape:", mcmc.chain[: mcmc._index].shape)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Parameter traces with fixed x-axis - only plot actual chain values
    active_chain = mcmc.chain[: mcmc._index]  # Only use actual samples
    for i, (param, label) in enumerate(zip(active_chain.T, ["r₁", "r₂", "D"])):
        ax1.plot(np.arange(len(param)), param, label=label)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Parameter Value")
    ax1.set_title("Parameter Traces")
    ax1.legend()

    # Data and fit comparison
    ax2.plot(x_points, noisy_data, "k.", alpha=0.5, label="Data")
    ax2.plot(x_points, true_solution, "g-", label="True")

    final_params = active_chain[-1]  # Use last actual sample
    print("\nFinal parameters:", final_params)
    final_solution = target.solve_steady_state(final_params)
    print("Final solution exists:", final_solution is not None)
    if final_solution is not None:
        print("Final solution range:", np.min(final_solution), np.max(final_solution))
        ax2.plot(x_points, final_solution, "r--", label="MCMC Fit")

    ax2.set_xlabel("x")
    ax2.set_ylabel("u(x)")
    ax2.set_title("Data and Model Fit")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print results using chain values
    print("\nResults:")
    print("True parameters:", true_params)
    print("Estimated parameters (mean of last 100 samples):")
    print(np.mean(active_chain[-100:], axis=0))
    print("\nParameter standard deviations:")
    print(np.std(active_chain[-100:], axis=0))

    plt.hist(
        mcmc.chain[: mcmc._index],
        bins=30,
        density=True,
        histtype="step",
        label=["r1", "r2", "D"],
    )
    plt.title("Parameter Distribution after Sampling")
    plt.xlabel("Parameter Value")
    plt.legend()
    plt.show()
