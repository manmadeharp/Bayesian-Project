# karhunen_expansion Monte Carlo up to N = 100 
# computing what the function looks like over a range of x for each given sample
# karhunen_expansion Monte Carlo up to N = 100 
# computing what the function looks like over a range of x for each given sample


# import numpy as np
# import matplotlib.pyplot as plt
# import scipy
#
# rg = np.random.default_rng(42)
#
# def karhunen_expansion(x, k_start, k_end, alpha):
#     # Precompute k * np.pi once and use it for eigenvalue and eigenfunction computation
#     k_pi = np.arange(k_start, k_end) * np.pi
#     
#     # Vectorized computation of eigenvalues (k * pi) ^ alpha
#     eigenvalue = np.power(k_pi, -alpha)
#     
#     # Compute the eigenfunctions sin(k * pi * x) for each x
#     # We need to expand k_pi and x to align their dimensions for element-wise multiplication
#     eigenfunction = np.sin(k_pi[:, None] * x)  # Shape (k, len(x))
#     
#     # Generate k normal random variables
#     rn_k = rg.normal(0, 1, k_end - k_start)
#     
#     # Compute the Karhunen-Loève expansion for each x
#     return np.sum(eigenvalue[:, None] * eigenfunction * rn_k[:, None], axis=0)
#
# # # Testing the function with k=100 and x=10 values
# # k = 100000
# # x = np.linspace(0, 1, 10000)
# #
# # #print(karhunen_expansion(x, k, 1))
# # alpha = 1
# # y = karhunen_expansion(x, k, alpha)
# #
# # # Plotting the results
# # plt.plot(x, y, label=f'Karhunen Expansion (k={k}, alpha={alpha})')
# # plt.title('Karhunen-Loève Expansion')
# # plt.xlabel('x')
# # plt.ylabel('Expansion Value')
# # plt.grid(True)
# # plt.legend()
# # plt.show()
#
#
#
# k_values = [10000, 20000, 30000]  # Different k values to explore
# # alpha_values = np.linspace(1, 8, 8)  # 8 different alpha values between 0.1 and 2
# alpha_values = [1, 2, 3, 4, 5, 6, 7, 8]
#
# n_points = 3000  # Number of x points as per experiment
# x = np.linspace(0, 1, n_points, endpoint=False)  # x_i = i/3000 Range for x values (more points for higher resolution)
#
# # Loop over each alpha value and generate individual plot windows
# for j, alpha in enumerate(alpha_values):
#     plt.figure()  # Create a new figure window for each alpha
#     # Plot the Karhunen-Loève expansion for all k values for this alpha
#     y = karhunen_expansion(x, 1, 10000, alpha)
#     plt.plot(x, y, label='k=100000')
#     for k in range(1, 8):
#         y += karhunen_expansion(x, k*10000, (k+1)*10000, alpha)
#         plt.plot(x, y, label=f'k={k*10000}')
#     
#     # Set title, labels, and grid for the current plot
#     plt.title(f'Karhunen-Loève Expansion (α={alpha:.2f})')
#     plt.xlabel('x')
#     plt.ylabel('Expansion Value')
#     plt.grid(True)
#     plt.legend()
#
#     # Show the current figure
#     plt.show()  # Display the current plot window


import numpy as np
import matplotlib.pyplot as plt

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

def compute_mode_statistics(k_max, alpha, num_samples=1000):
    """
    Compute empirical mean and variance of lambda_k * xi_k using stored coefficients.

    Parameters:
    - k_max (int): Maximum k value to consider (e.g., 300000).
    - alpha (float): Parameter controlling eigenvalue decay.
    - num_samples (int): Number of samples to estimate statistics.

    Returns:
    - k_vals (ndarray): Sampled k indices.
    - emp_means (ndarray): Empirical means of lambda_k * xi_k.
    - emp_vars (ndarray): Empirical variances of lambda_k * xi_k.
    - theo_vars (ndarray): Theoretical variances (lambda_k^2).
    """
    # Sample k indices every 10,000 up to k_max
    k_indices = np.arange(10000 - 1, k_max, 10000, dtype=int)  # Adjust for 0-based indexing
    num_k = len(k_indices)
    
    # Compute lambda_k and xi_k for the full range once
    _ = karhunen_expansion(np.linspace(0, 1, 3000), 1, k_max, alpha)  # Trigger coefficient storage
    lambda_k = karhunen_expansion.lambda_k
    xi_k = karhunen_expansion.xi_k
    
    # Select sampled k values and corresponding coefficients
    sampled_lambda_k = lambda_k[k_indices]
    sampled_xi_k = xi_k[k_indices]
    
    # Generate num_samples realizations by resampling xi_k
    xi_k_samples = rg.standard_normal((num_samples, num_k))
    mode_samples = sampled_lambda_k * xi_k_samples  # Shape (num_samples, num_k)
    
    # Compute empirical means and variances
    emp_means = np.mean(mode_samples, axis=0)
    emp_vars = np.var(mode_samples, axis=0)
    
    # Compute theoretical variances
    theo_vars = sampled_lambda_k ** 2
    
    return k_indices + 1, emp_means, emp_vars, theo_vars  # Return 1-based k values

# Define experiment parameters
k_max = 100000  # Maximum truncation level
alpha_values = [1, 2, 3, 4, 5, 6, 7, 8]  # Alpha values as specified
num_samples = 1000  # Number of samples for empirical statistics

# Create a 2x4 subplot grid for all alpha values
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten for easy indexing

# Loop over each alpha value and plot in the corresponding subplot
for idx, alpha in enumerate(alpha_values):
    ax = axes[idx]
    k_vals, emp_means, emp_vars, theo_vars = compute_mode_statistics(k_max, alpha, num_samples)
    
    # Plot empirical variances
    ax.scatter(k_vals, emp_vars, label='Empirical Variance', color='blue', alpha=0.6)
    # Plot theoretical variances
    ax.plot(k_vals, theo_vars, label='Theoretical Variance', color='red', linestyle='--')
    # Plot empirical means (should be near zero)
    ax.scatter(k_vals, emp_means, label='Empirical Mean', color='green', alpha=0.6)
    
    # Set title, labels, and grid
    ax.set_title(f'α={alpha}')
    ax.set_xlabel('k')
    ax.set_ylabel('Variance / Mean')
    ax.set_yscale('log')  # Log scale for better visibility of variance decay
    ax.grid(True)
    ax.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()

# Optional: Generate and display sample paths for verification
for alpha in alpha_values:
    plt.figure()
    y1 = karhunen_expansion(np.linspace(0, 1, 3000), 1, 100000, alpha)
    y2 = karhunen_expansion(np.linspace(0, 1, 3000), 1, 200000, alpha)
    y3 = karhunen_expansion(np.linspace(0, 1, 3000), 1, 300000, alpha)
    plt.plot(y1, label='k=100000')
    plt.plot(y2, label='k=200000')
    plt.plot(y3, label='k=300000')
    plt.title(f'Karhunen-Loève Expansion (α={alpha:.2f})')
    plt.xlabel('x index')
    plt.ylabel('Expansion Value')
    plt.grid(True)
    plt.legend()
    plt.show()
