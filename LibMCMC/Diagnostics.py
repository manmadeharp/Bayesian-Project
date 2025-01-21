from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde

from MetropolisHastings import MetropolisHastings


class MCMCDiagnostics:
    """Comprehensive MCMC diagnostics and visualization"""

    def __init__(self, sampler: MetropolisHastings, true_value):
        self.sampler = sampler
        self.chain = sampler.chain[: sampler._index]
        self.n_samples = sampler._index
        self.n_params = self.chain.shape[1]
        self.diagnostics = self._calculate_diagnostics()

    # Core statistical computations
    def _compute_ess(self, x: np.ndarray) -> float:
        """Compute effective sample size for a parameter chain"""
        n = len(x)
        if n <= 1:
            return 0
        acf = np.correlate(x - np.mean(x), x - np.mean(x), mode="full")[n - 1 :]
        acf = acf / acf[0]
        cutoff = np.where((acf < 0) | (acf < 0.05))[0]
        cutoff = cutoff[0] if len(cutoff) > 0 else len(acf)
        ess = n / (1 + 2 * np.sum(acf[1:cutoff]))
        return max(1, ess)

    def _compute_act(self, x: np.ndarray) -> float:
        """Compute integrated autocorrelation time"""
        ess = self._compute_ess(x)
        return len(x) / ess if ess > 0 else np.inf

    def _compute_autocorr(
        self, param_idx: int, max_lag: Optional[int] = None
    ) -> np.ndarray:
        """Compute autocorrelation for a parameter"""
        if max_lag is None:
            max_lag = min(100, self.n_samples // 5)

        acf = []
        x = self.chain[:, param_idx]
        for k in range(max_lag):
            if len(x[k:]) > 0 and len(x[:-k]) > 0:
                correlation = np.corrcoef(x[:-k], x[k:])[0, 1]
                acf.append(correlation)
        return np.array(acf)

    def _calculate_diagnostics(self) -> Dict:
        """Calculate all MCMC diagnostics"""
        # Basic statistics
        mean_estimate = np.mean(self.chain, axis=0)
        cov_estimate = np.cov(self.chain.T)

        # Advanced statistics
        ess = np.array(
            [self._compute_ess(self.chain[:, i]) for i in range(self.n_params)]
        )
        act = np.array(
            [self._compute_act(self.chain[:, i]) for i in range(self.n_params)]
        )

        # R-hat calculation
        n_split = self.n_samples // 2
        chains = [self.chain[:n_split], self.chain[n_split:]]
        within_var = np.mean([np.var(c, axis=0) for c in chains], axis=0)
        chain_means = np.array([np.mean(c, axis=0) for c in chains])
        between_var = np.var(chain_means, axis=0) * n_split
        r_hat = np.sqrt((within_var + between_var / n_split) / within_var)

        return {
            "mean_estimate": mean_estimate,
            "covariance_estimate": cov_estimate,
            "ess": ess,
            "act": act,
            "r_hat": r_hat,
            "acceptance_rate": self.sampler.acceptance_count / self.n_samples,
            "n_samples": self.n_samples,
        }

    # Plotting methods
    def _plot_trajectory(self, ax: plt.Axes):
        """Plot chain trajectory"""
        ax.plot(self.chain[:, 0], self.chain[:, 1], "k.", alpha=0.1, markersize=1)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("Chain Trajectory")

    def _plot_traces(self, ax: plt.Axes):
        """Plot trace plots with running means"""
        for i in range(self.n_params):
            ax.plot(self.chain[:, i], label=f"X{i+1}", alpha=0.5)
            running_mean = np.cumsum(self.chain[:, i]) / np.arange(
                1, self.n_samples + 1
            )
            ax.plot(running_mean, "--", label=f"Mean X{i+1}")
        ax.set_xlabel("Iteration")
        ax.set_title("Trace Plots")
        ax.legend()

    def _plot_autocorr(self, ax: plt.Axes):
        """Plot autocorrelation"""
        max_lag = min(100, self.n_samples // 5)
        for i in range(self.n_params):
            acf = self._compute_autocorr(i, max_lag)
            ax.plot(acf, label=f"X{i+1}")
        ax.set_title("Autocorrelation")
        ax.legend()

    def _plot_acceptance_rate(self, ax: plt.Axes):
        """Plot running acceptance rate"""
        #        window = min(500, self.n_samples // 20)
        #        acc_rates = [np.mean(np.diff(self.chain[i:i+window, 0]) != 0)
        #                    for i in range(0, self.n_samples-window, window)]
        #        ax.plot(range(0, self.n_samples-window, window), acc_rates)
        ax.plot(
            range(0, self.n_samples), self.sampler.acceptance_rates[: self.n_samples]
        )
        ax.axhline(y=0.234, color="r", linestyle="--", label="Optimal")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Acceptance Rate")
        ax.set_title("Running Acceptance Rate")
        ax.legend()

    def _plot_ess(self, ax: plt.Axes):
        """Plot effective sample size"""
        ax.bar([f"X{i+1}" for i in range(self.n_params)], self.diagnostics["ess"])
        ax.set_ylabel("ESS")
        ax.set_title("Effective Sample Size")

    def _plot_marginals(self, ax: plt.Axes):
        """Plot marginal distributions"""
        for i in range(self.n_params):
            ax.hist(self.chain[:, i], bins=50, density=True, alpha=0.5, label=f"X{i+1}")
        ax.set_title("Marginal Distributions")
        ax.legend()

    # Main public methods
    def plot_diagnostics(self, show: bool = True) -> Tuple[plt.Figure, np.ndarray]:
        """Create comprehensive diagnostic plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        self._plot_trajectory(axes[0, 0])
        self._plot_traces(axes[0, 1])
        self._plot_autocorr(axes[0, 2])
        self._plot_acceptance_rate(axes[1, 0])
        self._plot_ess(axes[1, 1])
        self._plot_marginals(axes[1, 2])
        plt.tight_layout()
        fig.savefig(f"./Plots/MCMC_plot_{type(self.sampler).__name__}")
        if show:
            plt.show()
        return fig, axes

    def print_summary(self):
        """Print comprehensive MCMC diagnostics summary"""
        print("\nMCMC Summary Statistics:")
        print("=" * 50)

        print("\nConvergence Diagnostics:")
        print(f"R-hat: {self.diagnostics['r_hat']}")
        print(f"Effective Sample Size: {self.diagnostics['ess']}")
        print(f"Integrated ACT: {self.diagnostics['act']}")

        print("\nSampling Statistics:")
        print(f"Final acceptance rate: {self.diagnostics['acceptance_rate']:.3f}")
        print(f"Number of samples: {self.diagnostics['n_samples']}")

        print("\nParameter Estimates:")
        print("Means:")
        print(f"Estimated: {self.diagnostics['mean_estimate']}")
        print("\nCovariance:")
        print(self.diagnostics["covariance_estimate"])

    def plot_target_distribution(self):
        """Plot an approximation of the target distribution with KDE and MCMC samples"""
        # Extract chain samples
        x_samples, y_samples = self.chain[:, 0], self.chain[:, 1]

        # Compute KDE on the samples
        kde = gaussian_kde(self.chain.T)

        # Determine plot ranges with padding
        x_min, x_max = x_samples.min(), x_samples.max()
        y_min, y_max = y_samples.min(), y_samples.max()
        x_pad = 0.1 * (x_max - x_min)
        y_pad = 0.1 * (y_max - y_min)

        # Create grid for plotting
        x = np.linspace(x_min - x_pad, x_max + x_pad, 500)
        y = np.linspace(y_min - y_pad, y_max + y_pad, 500)
        x1, x2 = np.meshgrid(x, y)
        pos = np.vstack([x1.ravel(), x2.ravel()])

        # Evaluate KDE on the grid
        Z = kde(pos).reshape(x1.shape)

        # Plot KDE contours
        plt.figure(figsize=(10, 8))
        plt.contourf(x1, x2, Z, levels=30, cmap="viridis", alpha=0.8)
        plt.colorbar(label="Density")

        # Overlay MCMC samples
        plt.scatter(
            x_samples, y_samples, alpha=0.1, color="k", s=1, label="MCMC samples"
        )

        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")

        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Target Distribution Approximation with MCMC Samples")
        plt.legend()
        plt.show()
