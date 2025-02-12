import numpy as np
import matplotlib.pyplot as plt

# Set up parameters
k_max = 100  # Maximum wavenumber to analyze
alphas = [4, 5, 6, 7, 8]  # Values of α to test
k_values = np.arange(1, k_max + 1)  # Wavenumbers (start at 1 to avoid division by zero)

# Create figure with log-log plot
plt.figure(figsize=(10, 6))
plt.title('Spectral Decay Analysis for Fractional Laplacian Eigenproblem')
plt.xlabel('log(k)')
plt.ylabel('log|û(k)|')

# Calculate and plot for each alpha
for alpha in alphas:
    # Compute Fourier coefficients
    u_hat = (k_values * np.pi)**(-alpha/2)
    
    # Take logarithms
    log_k = np.log(k_values)
    log_u = np.log(u_hat)
    
    # Fit linear regression to get decay rate
    slope, intercept = np.polyfit(log_k, log_u, 1)
    
    # Plot results
    plt.plot(log_k, log_u, 
             label=rf'$\alpha={alpha}$ (slope: {slope:.2f})')

# Add reference line for C² smoothness requirement (slope = -3)
# Using α=6 as reference point since (6/2=3)
ref_slope = -3
x_ref = np.array([min(log_k), max(log_k)])
y_ref = ref_slope * x_ref + np.log(np.pi**(-6/2))  # Match intercept for α=6
plt.plot(x_ref, y_ref, 'k--', label='C² smoothness threshold (slope = -3)')

plt.legend()
plt.grid(True)
plt.tight_layout()

# Calculate and print critical values
print("Critical slope analysis:")
for alpha in alphas:
    theoretical_slope = -alpha/2
    print(f"α = {alpha}: Theoretical slope = {theoretical_slope:.1f} "
          f"({'≥ -3' if theoretical_slope >= -3 else '< -3'})")

plt.show()
