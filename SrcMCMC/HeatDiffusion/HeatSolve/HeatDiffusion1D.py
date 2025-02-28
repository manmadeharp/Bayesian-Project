import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
from LibMCMC.Distributions import Proposal, TargetDistribution
from LibMCMC.MetropolisHastings import MetropolisHastings
from LibMCMC.PRNG import RNG
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve


@dataclass
class DirichletHeatConfig:
    """Configuration for heat equation with Dirichlet BCs"""

    L: float = 1.0  # Domain length
    T: float = 1.0  # Final time
    nx: int = 100  # Number of spatial points
    nt: int = 100  # Number of time points
    left_bc: Callable = lambda t: 0  # Left boundary condition
    right_bc: Callable = lambda t: 0  # Right boundary condition


class DirichletHeatSolver:
    """
    Solver for heat equation with Dirichlet boundary conditions using
    Crank-Nicolson scheme
    """

    def __init__(self, config: DirichletHeatConfig):
        self.config = config

        # Setup grid
        self.dx = config.L / (config.nx - 1)
        self.dt = config.T / config.nt
        self.x = np.linspace(0, config.L, config.nx)
        self.t = np.linspace(0, config.T, config.nt)

        # Stability parameter
        self.r = self.dt / (self.dx**2)
        if self.r > 0.5:
            print(f"Warning: Grid Fourier number {self.r} > 0.5")
            print("Solution may be unstable")

        # Setup Crank-Nicolson matrices
        self._setup_matrices()

    def _setup_matrices(self):
        """Setup matrices for Crank-Nicolson scheme"""
        nx = self.config.nx
        r = self.r

        # Interior points only (nx-2 points)
        main_diag = (1 + r) * np.ones(nx - 2)
        off_diag = -0.5 * r * np.ones(nx - 3)

        # LHS matrix (implicit part)
        self.A = diags(
           [off_diag, main_diag, off_diag],
            [-1, 0, 1],
            shape=(nx - 2, nx - 2),
            format="csc",
        )

        # RHS matrix (explicit part)
        self.B = diags(
            [-off_diag, (2 - main_diag), -off_diag],
            [-1, 0, 1],
            shape=(nx - 2, nx - 2),
            format="csc",
        )

    def solve(self, initial_condition: np.ndarray) -> np.ndarray:
        """
        Solve the heat equation

        Args:
            initial_condition: Initial temperature distribution

        Returns:
            u: Solution array of shape (nt, nx)
        """
        nx, nt = self.config.nx, self.config.nt
        u = np.zeros((nt, nx))
        r = self.r

        # Set initial condition
        u[0] = initial_condition

        # Time stepping
        for k in range(nt - 1):
            # Current time
            t_now = self.t[k]
            t_next = self.t[k + 1]

            # Get interior points
            interior = u[k, 1:-1]

            # RHS vector
            b = self.B @ interior

            # Add boundary contributions
            # Left boundary
            b[0] += 0.5 * r * (self.config.left_bc(t_now) + self.config.left_bc(t_next))

            # Right boundary
            b[-1] += (
                0.5 * r * (self.config.right_bc(t_now) + self.config.right_bc(t_next))
            )

            # Solve for interior points
            u[k + 1, 1:-1] = spsolve(self.A, b)

            # Update boundary values
            u[k + 1, 0] = self.config.left_bc(t_next)
            u[k + 1, -1] = self.config.right_bc(t_next)

        return u

    def analytical_solution(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute analytical solution u(x,t) = sin(pix)exp(-pi^2t)"""
        X, T = np.meshgrid(x, t)
        return np.sin(np.pi * X) * np.exp(-(np.pi**2) * T)

    def compute_error(
        self, numerical: np.ndarray, analytical: np.ndarray
    ) -> np.ndarray:
        """Compute absolute error between numerical and analytical solutions"""
        return np.abs(numerical - analytical)


class DirichletHeatInverse(TargetDistribution):
    """
    Inverse problem for heat equation with Dirichlet BCs
    """

    def __init__(
        self,
        solver: DirichletHeatSolver,
        observations: np.ndarray,
        observation_times: np.ndarray,
        observation_locs: np.ndarray,
        sigma: float,
        prior_mean: Optional[np.ndarray] = None,
        prior_std: Optional[float] = None,
    ):
        self.sigma = sigma
        super().__init__(
            prior=sp.norm, likelihood=sp.norm, data=observations, sigma=sigma
        )
        self.solver = solver
        self.obs_times = observation_times
        self.obs_locs = observation_locs

        # Find indices for observation times and locations
        self.time_indices = [np.abs(solver.t - t).argmin() for t in observation_times]
        self.space_indices = [np.abs(solver.x - x).argmin() for x in observation_locs]

        # Set prior parameters
        nx = solver.config.nx
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(nx)
        self.prior_std = prior_std if prior_std is not None else np.ones(nx)

    def log_likelihood(self, x: np.ndarray) -> np.float64:
        """Compute log likelihood"""
        try:
            # Ensure boundary conditions match
            if not np.isclose(x[0], self.solver.config.left_bc(0)) or not np.isclose(
                x[-1], self.solver.config.right_bc(0)
            ):
                return np.float64(-np.inf)

            # Solve forward problem
            solution = self.solver.solve(x)

            # Extract solution at observation points
            predicted = solution[np.ix_(self.time_indices, self.space_indices)]

            # Compute log likelihood
            residuals = predicted - self.data
            return super().log_likelihood.logpdf(residuals, scale=self.sigma)
        except:
            return np.float64(-np.inf)

    def log_prior(self, x: np.ndarray) -> np.float64:
        """Compute log prior"""
        # Check boundary conditions
        if not np.isclose(x[0], self.solver.config.left_bc(0)) or not np.isclose(
            x[-1], self.solver.config.right_bc(0)
        ):
            return np.float64(-np.inf)

        # Compute prior only for interior points
        return np.sum(super().log_prior(x))


def test_dirichlet_heat():
    """Test heat equation solver with Dirichlet BCs"""

    # 1. Setup problem configuration
    config = DirichletHeatConfig(
        L=1.0,
        T=1.0,
        nx=50,
        nt=5000,
        left_bc=lambda t: 0,  # Zero boundary conditions
        right_bc=lambda t: 0,
    )

    solver = DirichletHeatSolver(config)

    # 2. Generate synthetic data
    # True initial condition (satisfy boundary conditions)
    x = solver.x
    true_ic = np.sin(np.pi * x)  # Satisfies zero BCs

    # Solve forward problem
    true_solution = solver.solve(true_ic)

    # Create observations
    noise_std = 0.05
    obs_times = np.linspace(0, config.T, 10)
    obs_locs = np.linspace(0, config.L, 20)[1:-1]  # Exclude boundaries

    # Generate noisy observations
    obs_t_idx = [np.abs(solver.t - t).argmin() for t in obs_times]
    obs_x_idx = [np.abs(solver.x - x).argmin() for x in obs_locs]

    observations = true_solution[
        np.ix_(obs_t_idx, obs_x_idx)
    ] + noise_std * np.random.randn(len(obs_times), len(obs_locs))

    # 3. Setup inverse problem
    target = DirichletHeatInverse(
        solver=solver,
        observations=observations,
        observation_times=obs_times,
        observation_locs=obs_locs,
        sigma=noise_std,
        prior_mean=np.zeros_like(x),
        prior_std=1.0,
    )

    # 4. Setup MCMC
    # Initial guess (satisfying BCs)
    initial_state = 0.5 * np.sin(2 * np.pi * x)

    # Proposal (only varies interior points)
    proposal_scale = 0.1 * np.eye(config.nx - 2)  # For interior points
    proposal = Proposal(sp.multivariate_normal, scale=proposal_scale)

    # Modified proposal to handle boundary conditions
    def propose_with_bcs(current):
        # Propose new interior points
        proposed_interior = proposal.propose(current[1:-1])
        # Keep boundary values fixed
        proposed = np.zeros_like(current)
        proposed[0] = config.left_bc(0)
        proposed[-1] = config.right_bc(0)
        proposed[1:-1] = proposed_interior
        return proposed

    proposes = Proposal(sp.norm, 5)

    # 5. Run MCMC
    mcmc = MetropolisHastings(target, proposes, initial_state)
    mcmc(5000)

    # 6. Plot results
    plt.figure(figsize=(15, 5))

    # Plot initial condition reconstruction
    plt.subplot(121)
    plt.plot(x, true_ic, "k-", label="True")
    plt.plot(x, mcmc.chain[-1000:].mean(axis=0), "r--", label="Posterior Mean")
    plt.fill_between(
        x,
        np.percentile(mcmc.chain[-1000:], 5, axis=0),
        np.percentile(mcmc.chain[-1000:], 95, axis=0),
        color="r",
        alpha=0.2,
        label="90% Credible Interval",
    )
    plt.legend()
    plt.title("Initial Condition Recovery")

    # Plot observations and predictions
    plt.subplot(122)
    plt.plot(x, true_solution.T, "k-", alpha=0.1)
    plt.scatter(
        obs_locs.repeat(len(obs_times)),
        observations.flatten(),
        c="r",
        alpha=0.2,
        s=10,
        label="Observations",
    )

    # Plot some predictions from posterior
    for ic in mcmc.chain[-1000::100]:
        pred = solver.solve(ic)
        plt.plot(x, pred.T, "b-", alpha=0.1)

    plt.legend()
    plt.title("Observations and Predictions")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_dirichlet_heat()
