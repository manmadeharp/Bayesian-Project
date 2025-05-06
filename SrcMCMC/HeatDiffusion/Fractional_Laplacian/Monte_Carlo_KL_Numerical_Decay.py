import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import math

plt.rcParams.update({'font.size': 17})  

# Initialize random number generator with a seed for reproducibility
rg = np.random.default_rng(42)

# Initialize function object with attributes
def karhunen_expansion(x, k_start, k_end, alpha):
    # Precompute k * np.pi once and use it for eigenvalue and eigenfunction computation
    k_pi = np.arange(k_start, k_end) * np.pi
    
    # Vectorized computation of eigenvalues (k * pi) ^ (-alpha) to match the experiment
    eigenvalue = np.power(k_pi, -alpha)
    karhunen_expansion.lambda_k = eigenvalue  # Attach eigenvalues as a function attribute
    
    # Compute the eigenfunctions sin(k * pi * x) for each x
    eigenfunction = np.sin(k_pi[:, None] * x)  # Shape (k, len(x))
    
    # Generate k normal random variables and attach as a function attribute
    rn_k = rg.normal(0, 1, k_end - k_start)
    karhunen_expansion.xi_k = rn_k
    
    # Compute the Karhunen-Loève expansion for each x
    return np.sum(eigenvalue[:, None] * eigenfunction * rn_k[:, None], axis=0)

def compute_mode_statistics(k_max, alpha, num_samples=1000, 
                            x=None, k_sample_strategy="uniform", 
                            num_k=20, percentile=None):
    """
    Compute empirical and theoretical statistics for KL expansion modes.

    Parameters:
    - k_max (int): Maximum truncation level
    - alpha (float): Decay exponent for eigenvalues
    - num_samples (int): Number of Monte Carlo samples
    - x (array): Spatial grid (if needed for KL storage)
    - k_sample_strategy (str): "uniform" or "percentile"
    - num_k (int): Number of k-values to sample
    - percentile (tuple): (low_percent, high_percent) if using percentile sampling

    Returns:
    - k_vals: 1-based k indices used
    - emp_means, emp_vars: empirical stats
    - theo_vars: λ_k^–α
    """
    if x is None:
        x = np.linspace(0, 1, 1000)

    # Trigger computation and store lambda_k
    _ = karhunen_expansion(x, 1, k_max + 1, alpha)
    lambda_k_full = karhunen_expansion.lambda_k  # length k_max

    # Choose which k indices to use
    
    if k_sample_strategy == "uniform":
        k_indices = np.round(np.linspace(0, k_max - 1, num=num_k)).astype(int)
        k_indices = np.unique(k_indices)  # Prevent duplicates due to rounding

    elif k_sample_strategy == "percentile":
        assert percentile is not None and len(percentile) == 2
        low, high = percentile
        lo_idx = int(low / 100 * k_max)
        hi_idx = int(high / 100 * k_max)
        k_indices = np.linspace(lo_idx, hi_idx - 1, num=num_k, dtype=int)
        k_indices = np.unique(k_indices)
    else:
        raise ValueError("Unknown k_sample_strategy")

    sampled_lambda_k = lambda_k_full[k_indices]

    # Draw samples from standard normal
    xi_k_samples = np.random.randn(num_samples, len(k_indices))
    mode_samples = sampled_lambda_k * xi_k_samples

    emp_means = np.mean(mode_samples, axis=0)
    emp_vars = np.var(mode_samples, axis=0)
    theo_vars = sampled_lambda_k ** 2  # Since Var(aZ) = a^2 Var(Z)

    return k_indices + 1, emp_means, emp_vars, theo_vars

# def plot_statistics(alpha_values, k_max=1000, num_samples=1000, 
#                     k_sample_strategy="uniform", num_k=20, percentile=None):
#     fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
#     axes = axes.flatten()
#
#     for idx, alpha in enumerate(alpha_values):
#         ax = axes[idx]
#         k_vals, emp_means, emp_vars, theo_vars = compute_mode_statistics(
#             k_max, alpha, num_samples=num_samples, 
#             k_sample_strategy=k_sample_strategy, num_k=num_k, percentile=percentile
#         )
#
#         ax.scatter(k_vals, emp_vars, label="Empirical Var", color='blue', alpha=0.6)
#         ax.plot(k_vals, theo_vars, label="Theoretical Var", color='red', linestyle='--')
#         ax.scatter(k_vals, emp_means, label="Empirical Mean", color='green', alpha=0.6)
#
#         ax.set_title(f"α = {alpha}")
#         ax.set_xlabel("k")
#         ax.set_ylabel("Variance / Mean")
#         ax.set_yscale('log')
#         ax.grid(True)
#         ax.legend()
#
#     plt.tight_layout()
#     plt.show()

def plot_statistics(alpha_values, k_max=1000, num_samples=1000, 
                    k_sample_strategy="uniform", num_k=20, percentile=None):
    num_alphas = len(alpha_values)
    num_cols = math.ceil(math.sqrt(num_alphas))
    num_rows = math.ceil(num_alphas / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)
    axes = axes.flatten()

    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx]
        k_vals, _, _, _ = compute_mode_statistics(
            k_max, alpha, num_samples=num_samples,
            k_sample_strategy=k_sample_strategy, num_k=num_k, percentile=percentile
        )

        # For boxplots, regenerate raw samples
        x = np.linspace(0, 1, 1000)
        _ = karhunen_expansion(x, 1, k_max + 1, alpha)
        lambda_k = karhunen_expansion.lambda_k

        if k_sample_strategy == "uniform":
            k_indices = np.round(np.linspace(0, k_max - 1, num=num_k)).astype(int)
        elif k_sample_strategy == "percentile":
            low, high = percentile
            lo_idx = int(low / 100 * k_max)
            hi_idx = int(high / 100 * k_max)
            k_indices = np.round(np.linspace(lo_idx, hi_idx - 1, num=num_k)).astype(int)
        k_indices = np.unique(k_indices)
        sampled_lambda_k = lambda_k[k_indices]

        # Generate raw samples
        samples = sampled_lambda_k * np.random.randn(num_samples, len(k_indices))

        # Create box plot per k
        
        ax.boxplot(samples, positions=k_vals, widths=0.6, patch_artist=True,
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1),
                capprops=dict(linewidth=1),
                medianprops=dict(linewidth=1),
                flierprops=dict(marker='o', markersize=3, linestyle='none', color='black'))

        ax.set_title(f"α = {alpha}", fontsize=14)
        ax.set_xlabel("k index")
        ax.set_ylabel("λ_k^α ξ_k samples")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_yscale('symlog', linthresh=1e-10)  # or 1e-8, depending on alpha
        # ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(k_vals[0] - k_vals[-1]/100, k_vals[-1] + k_vals[-1]/100)
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs='auto', numticks=10))

    # Turn off empty subplots
    for j in range(num_alphas, num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # for alpha in alpha_values:
    #     upper = +alpha * np.sqrt(lambda_k[k]) * phi_k[k](x)
    #     lower = -alpha * np.sqrt(lambda_k[k]) * phi_k[k](x)
    #     plt.plot(x, upper, 'r--')
    #     plt.plot(x, lower, 'b--')

def plot_kl_envelopes(alpha_values, k_max=30000, num_samples=1000, gamma=3.0, img_height=300,
                      major_tick=1000, minor_tick=100, color=(0.2, 0.4, 1.0), gradient_power=6.0, step=1000):
    num_alphas = len(alpha_values)
    num_cols = math.ceil(math.sqrt(num_alphas))
    num_rows = math.ceil(num_alphas / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4.5 * num_rows), squeeze=False)
    axes = axes.flatten()
    
    # Initialize global y-range variables
    global_min = np.inf
    global_max = -np.inf

    # First pass: Calculate global min and max for consistent y-axis scaling
    for alpha in alpha_values:
        k_vals = np.arange(1, k_max + 1, step=step)  # Plot every nth mode
        lambda_k = (k_vals * np.pi) ** (-alpha)
        scale_k = np.sqrt(lambda_k)
        
        # Generate samples using correct dimensions
        samples = scale_k[:, None] * np.random.randn(len(k_vals), num_samples)
        
        means = samples.mean(axis=1)
        mins = samples.min(axis=1)
        maxs = samples.max(axis=1)

        global_min = min(global_min, mins.min())
        global_max = max(global_max, maxs.max())

    # Second pass: Create the plots
    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx]

        # KL sample generation (now with reduced k_vals)
        k_vals = np.arange(1, k_max + 1, step=step)  # Plot every nth mode
        lambda_k = (k_vals * np.pi) ** (-alpha)
        scale_k = np.sqrt(lambda_k)
        
        # Generate samples with correct dimensions
        samples = scale_k[:, None] * np.random.randn(len(k_vals), num_samples)
        
        means = samples.mean(axis=1)
        mins = samples.min(axis=1)
        maxs = samples.max(axis=1)

        # 2D image grid
        y = np.linspace(-1, 1, img_height).reshape(-1, 1)
        gradient = np.exp(-gamma * np.abs(y)**gradient_power)
        rgba = np.ones((img_height, len(k_vals), 4))
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = gradient

        # Image extent
        height = global_max - global_min  # Use global min/max
        x_extent = [k_vals[0], k_vals[-1]]  # Set x_extent based on k_vals range
        y_extent = [global_min, global_max]

        # Plot gradient
        ax.imshow(rgba, aspect='auto', extent=x_extent + y_extent, origin='lower',
                  interpolation='bilinear', zorder=1)
        
        # Mask outside ribbon
        ax.fill_between(k_vals, maxs, global_max + height * 0.1, color='white', zorder=2)
        ax.fill_between(k_vals, global_min - height * 0.1, mins, color='white', zorder=2)

        # Overlay curves
        ax.plot(k_vals, means, color='black', linewidth=2, label='Mean', zorder=3)
        ax.plot(k_vals, mins, color='red', linestyle='-', linewidth=1,  zorder=3)
        ax.plot(k_vals, maxs, color='red', linestyle='-', linewidth=1, zorder=3)

        # Aesthetics and ticks
        # ax.set_xticklabels([])  # Removes x-axis labels
        ax.set_xlim(k_vals[0], k_vals[-1])  # Adjust x-axis to match the range of k_vals
        ax.set_ylim(global_min - 0.05 * height, global_max + 0.05 * height)
        ax.set_title(f"α = {alpha}", fontsize=15)
        ax.set_xlabel("Mode index $k$" , fontsize=14)
        # ax.set_ylabel("KL Amplitude")
        ax.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
        ax.tick_params(axis='x', which='major', length=7, width=1.5)
        ax.tick_params(axis='x', which='minor', length=3, width=1)
        ax.tick_params(axis='y', labelsize=13)

        ax.tick_params(axis='x', which='major', rotation=45)
        ax.tick_params(axis='x', which='minor', rotation=45)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(False)
        ax.set_yscale('symlog', linthresh=1e-8)  # Apply symlog scaling to all subplots
        # ax.legend()

    plt.suptitle('Envelope of Modal Amplitudes of the Truncated Karhunen–Loève Expansion')
    fig.supylabel("Modal Amplitudes (Symbolic Logarithmic Scale)")
    # Hide unused axes
    for j in range(len(alpha_values), num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust the top to make room for the global title

    plt.subplots_adjust(hspace=0.2, wspace=0.2)  # Decrease the gap between plots by reducing hspace and wspace
    plt.show()

# Test the function again with large k_max and adjusted x-axis scaling
plot_kl_envelopes(
    alpha_values=[1.0, 2.0, 3.0, 4.0],
    k_max=300000,
    num_samples=10000,
    gamma=9.0,
    img_height=500,
    major_tick=10000,
    minor_tick=10000,
    step=100  # Plot every 1000th mode
)


def plot_sample_paths(alpha_values, k_vals, x=None):
    if x is None:
        x = np.linspace(0, 1, 1000)

    for alpha in alpha_values:
        plt.figure()
        for k_max in k_vals:
            y = karhunen_expansion(x, 1, k_max, alpha)
            plt.plot(x, y, label=f"k = {k_max}")
        plt.title(f"KL Expansion sample (α={alpha})")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend()
        plt.grid(True)
        plt.show()

# plot_statistics(
#     alpha_values=[1, 4],
#     k_max=10000,
#     num_samples=100,
#     k_sample_strategy="uniform",
#     num_k=15
# )

# plot_kl_envelopes(
#     alpha_values=[1.0, 5.0],
#     k_max=30000,
#     num_samples=10000,
#     gamma=4.0,
#     img_height=500,
#     major_tick=1000,
#     minor_tick=1000,
#     step=1000
# )
