from dataclasses import dataclass
from typing import Callable, List, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

np.random.seed(12)

plt.rcParams['text.usetex'] = True

plt.rcParams['font.family'] = 'serif'

# plt.rcParams['figure.dpi'] = 600

plt.rcParams['xtick.labelsize'] = 14  # x-axis tick label size
plt.rcParams['ytick.labelsize'] = 14  # y-axis tick label size

@dataclass
class HeatEquationConfig:
    """Configuration for heat equation solution."""

    L: float = 1.0  # Domain length
    T: float = 0.05  # Final time
    nx: int = 1000  # Number of spatial points
    nt: int = 1000  # Number of time points
    max_terms: int = 100  # Maximum number of Fourier terms
    k: float = 1.0  # Thermal diffusivity


class HeatEquationSolver:
    """Solver for the 1D heat equation with Dirichlet boundary conditions."""

    def __init__(self, config: HeatEquationConfig):
        """Initialize the solver with the given configuration."""
        self.config = config
        self.x = np.linspace(0, config.L, config.nx)
        self.t = np.linspace(0, config.T, config.nt)
        self.coefficients = None

    def set_coefficients(self, coefficients: np.ndarray) -> None:
        """Directly set the Fourier coefficients."""
        if len(coefficients) > self.config.max_terms:
            self.coefficients = coefficients[:self.config.max_terms]
        else:
            self.coefficients = coefficients
    
    def compute_initial_condition(self) -> np.ndarray:
        """Compute the initial condition from the coefficients."""
        if self.coefficients is None:
            raise ValueError("Coefficients must be set before computing initial condition")
            
        L = self.config.L
        x = self.x
        
        # Compute initial condition using the coefficients
        n_values = np.arange(1, len(self.coefficients) + 1)[:, np.newaxis]
        x_matrix = x[np.newaxis, :]
        sin_terms = np.sin(n_values * np.pi * x_matrix / L)
        
        # Sum up the contributions from all modes
        initial_condition = np.sum(self.coefficients[:, np.newaxis] * sin_terms, axis=0)
        
        return initial_condition

    def get_mode_values_at_time(self, t: float) -> np.ndarray:
        """Calculate the value of each Fourier mode at a specified time."""
        if self.coefficients is None:
            raise ValueError("Coefficients must be set first")

        L, k = self.config.L, self.config.k
        n_values = np.arange(1, len(self.coefficients) + 1)
        decay_factors = np.exp(-k * (n_values * np.pi / L) ** 2 * t)
        return self.coefficients * decay_factors

    def solve_at_time(self, t: float) -> np.ndarray:
        """Solve the heat equation at a specific time."""
        if self.coefficients is None:
            raise ValueError("Coefficients must be set first")
            
        L, k = self.config.L, self.config.k
        max_terms = len(self.coefficients)
        
        # Compute solution at time t
        n_values = np.arange(1, max_terms + 1)[:, np.newaxis]
        x_matrix = self.x[np.newaxis, :]

        sin_terms = np.sin(n_values * np.pi * x_matrix / L)
        exp_terms = np.exp(-k * (n_values * np.pi / L) ** 2 * t)
        mode_contributions = self.coefficients[:, np.newaxis] * sin_terms * exp_terms
        
        return np.sum(mode_contributions, axis=0)

    def solve_forward(self) -> np.ndarray:
        """Solve the forward heat equation at all time points."""
        if self.coefficients is None:
            raise ValueError("Coefficients must be set first")

        # Preallocate solution array
        solution = np.zeros((self.config.nt, self.config.nx))

        for i, t in enumerate(self.t):
            solution[i] = self.solve_at_time(t)

        return solution


def generate_random_coefficients(max_terms: int = 100, seed: int = None) -> tuple:
    """
    Generate two sets of random Fourier coefficients with the first coefficient identical.
    
    Args:
        max_terms: Number of Fourier terms to generate
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (coeffs1, coeffs2) with random values
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random coefficients
    coeffs1 = np.random.uniform(-5, 5, max_terms)
    coeffs2 = np.random.uniform(-5, 5, max_terms)
    
    # Make the first frequency identical between the two functions
    coeffs2[0] = coeffs1[0]
    
    return coeffs1, coeffs2

def create_initial_conditions() -> List[Callable]:
    """
    Create two different initial conditions with random sine waves.
    Functions will be zero at x=0 and x=1 to satisfy Dirichlet boundary conditions.
    
    Returns:
        List of callable functions for the initial conditions.
    """
    # Create random coefficients for 60 frequencies
    # np.random.seed(42)  # For reproducibility
    coeffs1 = np.random.uniform(-5, 5, 100)
    coeffs2 = np.random.uniform(-5, 5, 100)
    
    # Make the low frequencies similar between the two functions
    coeffs2[:1] = coeffs1[:1]
    
    def ic1(x):
        n_values = np.arange(1, 101)[:, np.newaxis]  # Shape (60, 1) for broadcasting
        sine_terms = np.sin(n_values * np.pi * x)   # Shape (60, len(x))
        return np.sum(coeffs1[:, np.newaxis] * sine_terms, axis=0)  # Sum along frequency axis
    
    def ic2(x):
        n_values = np.arange(1, 101)[:, np.newaxis]  # Shape (60, 1) for broadcasting
        sine_terms = np.sin(n_values * np.pi * x)   # Shape (60, len(x))
        return np.sum(coeffs2[:, np.newaxis] * sine_terms, axis=0)  # Sum along frequency axis
    
    return [ic1, ic2]


def compute_l2_norm_analytical(coeffs1: np.ndarray, coeffs2: np.ndarray, t: float, L: float = 1.0, k: float = 1.0) -> float:
    """
    Compute the L² norm of the difference between two solutions analytically.
    
    Uses the formula: ||u_g(·,t) - u_h(·,t)||_{L²} = sqrt(∑(A_n - B_n)²/2 · e^(-2(nπ)²t))
    
    Args:
        coeffs1: First set of Fourier coefficients
        coeffs2: Second set of Fourier coefficients
        t: Time at which to compute the norm
        L: Domain length
        k: Thermal diffusivity
    
    Returns:
        The L² norm of the difference
    """
    n_values = np.arange(1, len(coeffs1) + 1)
    diff_squared = (coeffs1 - coeffs2) ** 2
    exp_terms = np.exp(-2 * k * (n_values * np.pi / L) ** 2 * t)
    
    # The 1/2 factor comes from the orthogonality relation of sine functions
    return np.sqrt(np.sum(diff_squared * exp_terms / 2))


def visualize_heat_equation_evolution(
    config: HeatEquationConfig, num_time_points: int = 6
) -> None:
    """
    Visualize the evolution of two different initial conditions under the heat equation.

    Args:
        config: Configuration for the heat equation solver
        num_time_points: Number of time points to visualize
    """
    # Generate random coefficients
    coeffs1, coeffs2 = generate_random_coefficients(config.max_terms)
    
    # Create solvers
    solver1 = HeatEquationSolver(config)
    solver2 = HeatEquationSolver(config)
    
    # Set coefficients directly
    solver1.set_coefficients(coeffs1)
    solver2.set_coefficients(coeffs2)
    
    # Compute initial conditions for visualization
    ic1 = solver1.compute_initial_condition()
    ic2 = solver2.compute_initial_condition()
    
    # Solve forward problem
    solution1 = solver1.solve_forward()
    solution2 = solver2.solve_forward()
    
    # Use logarithmically spaced time points for better visualization
    if num_time_points > 1:
        # First point is t=0, rest are logarithmically spaced to catch early dynamics
        min_t = config.T/config.nt * 1.1  # Just above the first time step
        time_points = np.concatenate(([0], np.logspace(np.log10(min_t), np.log10(config.T), num_time_points-1)))
    else:
        time_points = np.array([0])  # Just initial condition if only one point requested

    # Find closest indices in the time array
    time_indices = [np.abs(solver1.t - t).argmin() for t in time_points]
    actual_times = [solver1.t[idx] for idx in time_indices]
    
    # Compute L2 norm analytically for all time points
    l2_diffs = [
        compute_l2_norm_analytical(coeffs1, coeffs2, t, config.L, config.k)
        for t in solver1.t
    ]

    # Create figure for initial conditions and coefficients
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot initial conditions
    axes[0].plot(solver1.x, ic1, "b-", label=r"$g(x)$")
    axes[0].plot(solver1.x, ic2, "r-", label=r"$h(x)$")
    axes[0].set_title(r"Initial Conditions at $t=0$")
    axes[0].set_xlabel(r"Position $x$", fontsize=16)
    axes[0].set_ylabel(r"$u(x,0)$", fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot Fourier coefficients - linear scale
    n_values = np.arange(1, config.max_terms + 1)
    axes[1].plot(n_values, solver1.coefficients, "b-", label=r"$g(x)$ Coefficients")
    axes[1].plot(n_values, solver2.coefficients, "r-", label=r"$h(x)$ Coefficients")
    axes[1].set_title(r"Fourier Coefficients of Initial Conditions")
    axes[1].set_xlabel(r"Fourier Frequency $n$", fontsize=16)
    axes[1].set_ylabel(r"$a_n$", fontsize=16)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(0, config.max_terms)

    values1 = solver1.get_mode_values_at_time(0)
    values2 = solver2.get_mode_values_at_time(0)
    max_abs_value = max(np.max(np.abs(values1)), np.max(np.abs(values2)))

    axes[1].set_ylim(-max_abs_value*1.1, max_abs_value*1.1)  # Symmetric around zero with padding

    plt.tight_layout()

    # Create figure for L2 difference
    plt.figure(figsize=(10, 6))
    plt.plot(solver1.t, l2_diffs)
    plt.title(r"L$^2$ Norm of Solution Difference", fontsize=19)
    plt.xlabel(r"Time $(t)$", fontsize=17)
    plt.ylabel(r"$\|u_1(\cdot,t) - u_2(\cdot,t)\|_2$", fontsize=17)
    plt.grid(True, alpha=0.3)

    # First graph loop (mode magnitudes)
    fig, axes = plt.subplots(num_time_points, 1, figsize=(12, 4 * num_time_points))
    if num_time_points == 1:
        axes = [axes]

    # Add a single title for the entire figure
    fig.suptitle(r"Dampened Fourier Coefficients Evolution Over Time", fontsize=18)
    fig.supylabel(r"$a_n \exp\left(-(n\pi)^2 t\right)$", fontsize=18)

    ymin = min(min(solution1[0]), min(solution2[0]))
    ymax = max(max(solution1[0]), max(solution2[0]))
    for i, t_idx in enumerate(time_indices):
        t = actual_times[i]
        
        # Get mode magnitudes at this time
        value1 = solver1.get_mode_values_at_time(t)
        value2 = solver2.get_mode_values_at_time(t)
        
        # Plot magnitudes with linear scaling
        ax = axes[i]
        ax.plot(n_values, value1, "b-")#, label=r"$g(x)$"  if i == 0 else None)
        ax.plot(n_values, value2, "r-")#, label=r"$h(x)$"  if i == 0 else None)
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)  # Add zero line

        # Add time as text annotation instead of title (1 is no go)
        ax.set_xlim(1, config.max_terms)
        axes[i].set_ylim(-6, 6)

        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.grid(True, alpha=0.3)

        
        ax.text(0.85, 0.02, f"$t = {t:.5f}$", transform=ax.transAxes, fontsize=15, fontweight='bold')

        if i == num_time_points - 1:
            ax.set_xlabel(r"Fourier Frequency $n$", fontsize=17)
            # ax.set_ylabel(r"$a_n \exp\left(-(n\pi)^2 t\right)$", fontsize=17)
        if i == num_time_points // 2:
            axes[i].legend(loc='lower right', frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom = 0.05, hspace=0.5, top=0.955)  # Increase hspace value

    # Second graph loop (solution evolution)
    fig, axes = plt.subplots(num_time_points, 1, figsize=(12, 4 * num_time_points))
    if num_time_points == 1:
        axes = [axes]

    # Add a single title for the entire figure
    fig.suptitle(r"Heat Equation Solution Evolution", fontsize=18)
    fig.supylabel(r"$u(x,t)$", fontsize=18)

    ymin = min(min(solution1[0]), min(solution2[0]))
    ymax = max(max(solution1[0]), max(solution2[0]))
    padding = (ymax - ymin) * 0.1  # Better padding calculation
    ymin = ymin - padding 
    ymax = ymax + padding

    for i, t_idx in enumerate(time_indices):
        t = actual_times[i]
        
        # Plot solutions
        axes[i].plot(solver1.x, solution1[t_idx], "b-") #, label=r"$g(x)$" if i == 0 else None
        axes[i].plot(solver1.x, solution2[t_idx], "r-" )#label=r"$h(x)$" if i == 0 else None)
        axes[i].text(0.87, 0.02, f"$t = {t:.5f}$", transform=axes[i].transAxes, fontsize=15, fontweight='bold')
        
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(ymin, ymax)
        axes[i].grid(True, alpha=0.3)
        
        if i == num_time_points - 1:
            axes[i].set_xlabel(r"Position $x$", fontsize=17)
        if i == 0:
            axes[i].legend(loc='lower right', frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom = 0.05, hspace=0.2, top=0.96)  # Increase hspace value


def main():
    """Main function to run visualization."""
    # Configure the heat equation solver
    config = HeatEquationConfig(
        L=1.0,  # Domain length
        T=0.10,  # Final time
        nx=5000,  # Spatial points
        nt=5000,  # Time points
        max_terms=100,  # Maximum Fourier terms
        k=1.0,  # Thermal diffusivity
    )

    # Run visualization
    visualize_heat_equation_evolution(config, num_time_points=6)

    # Show all figures

    # plt.rcdefaults()
    plt.show()


if __name__ == "__main__":
    main()
