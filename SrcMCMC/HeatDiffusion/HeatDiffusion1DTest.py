import matplotlib.pyplot as plt
import numpy as np
from HeatDiffusion1D import DirichletHeatConfig, DirichletHeatSolver


def test_heat_solver():
    """
    Test heat equation solver against analytical solution:
    u(x,t) = sin(πx)exp(-π²t)
    """
    # Setup configuration with better stability
    config = DirichletHeatConfig(
        L=1.0,
        T=1.0,
        nx=50,
        nt=5000,  # Increased time steps to reduce Fourier number
        left_bc=lambda t: 0,
        right_bc=lambda t: 0,
    )

    solver = DirichletHeatSolver(config)

    # Print grid parameters
    print(f"dx = {solver.dx:.6f}")
    print(f"dt = {solver.dt:.6f}")
    print(f"Fourier number = {solver.r:.6f}")

    # Initial condition
    x = solver.x
    initial_condition = np.sin(np.pi * x)

    # Solve numerically
    numerical_solution = solver.solve(initial_condition)

    # Compute analytical solution
    def analytical_solution(x, t):
        return np.sin(np.pi * x) * np.exp(-(np.pi**2) * t)

    X, T = np.meshgrid(x, solver.t)
    analytical = analytical_solution(X, T)

    # Compute error
    error = np.abs(numerical_solution - analytical)
    max_error = np.max(error)
    l2_error = np.sqrt(np.mean(error**2))

    print(f"Maximum error: {max_error:.2e}")
    print(f"L2 error: {l2_error:.2e}")

    # Create separate figures for better stability
    # Figure 1: Solutions at different times
    plt.figure(1)
    times = [0, 0.2, 0.5, 1.0]
    for t in times:
        tidx = np.abs(solver.t - t).argmin()
        plt.plot(x, numerical_solution[tidx], "--", label=f"t={t:.1f} (num)")
        plt.plot(x, analytical_solution(x, t), "k:", label=f"t={t:.1f} (exact)")
    plt.legend()
    plt.title("Solution at Different Times")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.grid(True)
    plt.savefig("Assets/heat_solutions.png")
    plt.close()

    # Figure 2: Error distribution
    plt.figure(2)
    plt.pcolormesh(X, T, error, shading="auto")
    plt.colorbar(label="Absolute Error")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Error Distribution")
    plt.savefig("Assets/heat_error_dist.png")
    plt.close()

    # Figure 3: Error evolution
    plt.figure(3)
    plt.semilogy(solver.t, np.max(error, axis=1), label="Max Error")
    plt.semilogy(solver.t, np.sqrt(np.mean(error**2, axis=1)), label="L2 Error")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.title("Error Evolution")
    plt.grid(True)
    plt.savefig("Assets/heat_error_evolution.png")
    plt.close()

    return numerical_solution, analytical, error


if __name__ == "__main__":
    numerical, analytical, error = test_heat_solver()
    print("\nTest completed.")
