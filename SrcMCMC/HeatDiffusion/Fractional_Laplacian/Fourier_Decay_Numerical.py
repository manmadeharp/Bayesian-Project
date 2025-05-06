# import numpy as np
# import matplotlib.pyplot as plt
#
# # Set up parameters
# k_max = 100  # Maximum wavenumber to analyze
# alphas = [4, 5, 6, 7, 8]  # Values of α to test
# k_values = np.arange(1, k_max + 1)  # Wavenumbers (start at 1 to avoid division by zero)
#
# # Create figure with log-log plot
# plt.figure(figsize=(10, 6))
# plt.title('Spectral Decay Analysis for Fractional Laplacian Eigenproblem')
# plt.xlabel('log(k)')
# plt.ylabel('log|û(k)|')
#
# # Calculate and plot for each alpha
# for alpha in alphas:
#     # Compute Fourier coefficients
#     u_hat = (k_values * np.pi)**(-alpha/2)
#     
#     # Take logarithms
#     log_k = np.log(k_values)
#     log_u = np.log(u_hat)
#     
#     # Fit linear regression to get decay rate
#     slope, intercept = np.polyfit(log_k, log_u, 1)
#     
#     # Plot results
#     plt.plot(log_k, log_u, 
#              label=rf'$\alpha={alpha}$ (slope: {slope:.2f})')
#
# # Add reference line for C² smoothness requirement (slope = -3)
# # Using α=6 as reference point since (6/2=3)
# ref_slope = -3
# x_ref = np.array([min(log_k), max(log_k)])
# y_ref = ref_slope * x_ref + np.log(np.pi**(-6/2))  # Match intercept for α=6
# plt.plot(x_ref, y_ref, 'k--', label='C² smoothness threshold (slope = -3)')
#
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
#
# # Calculate and print critical values
# print("Critical slope analysis:")
# for alpha in alphas:
#     theoretical_slope = -alpha/2
#     print(f"α = {alpha}: Theoretical slope = {theoretical_slope:.1f} "
#           f"({'≥ -3' if theoretical_slope >= -3 else '< -3'})")
#
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
#
# def generate_sample(x, K, alpha, rng):
#     """
#     Parameters:
#     - x: Array of spatial points.
#     - K: Number of terms in the expansion.
#     - alpha: Exponent of the covariance operator.
#     - phi_k: Array of K standard normal coefficients (N(0,1)).    
#     """
#     phi_k = rng.standard_normal(K)  # K random N(0,1) coefficients
#     sample = np.zeros_like(x)
#     for k in range(1, K+1):
#         sample += (k*np.pi)**(-alpha/2) * np.sin(k*np.pi*x) * phi_k[k-1]
#     return sample
#
# def spectral_derivative(u, order=2):
#     """Compute spectral derivative"""
#     N = len(u)
#     k = np.fft.fftfreq(N) * N
#     u_hat = np.fft.fft(u)
#     return np.fft.ifft((1j*k)**order * u_hat).real # 1j is the imaginary unit
#
# # Test parameters
# alpha_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # Lots of alpha values takes a long time to compute
# N_values = np.arange(32, 256, 32)
# num_samples = 100  # Number of samples per alpha
# K = 100000  # Number of terms in sum
#
# rng = np.random.default_rng(42)
# errors = {alpha: [] for alpha in alpha_values}
#
# for alpha in alpha_values:
#     alpha_errors = []
#     for N in N_values:
#         x = np.linspace(0, 1, N, endpoint=False)
#         sample_errors = []
#         
#         for _ in range(num_samples):
#             # Generate sample
#             u = generate_sample(x, K, alpha, rng)
#             
#             # Compute 2nd derivative
#             u_dd_spectral = spectral_derivative(u, order=2)
#             
#             # Compute "true" derivative with finer grid
#             N_fine = 2*N
#             x_fine = np.linspace(0, 1, N_fine, endpoint=False)
#             u_fine = generate_sample(x_fine, K, alpha, rng)
#             u_dd_true = spectral_derivative(u_fine, order=2)
#             u_dd_true = u_dd_true[::2]  # Downsample to match N
#             
#             # Compute error
#             error = np.max(np.abs(u_dd_spectral - u_dd_true))
#             sample_errors.append(error)
#             
#         alpha_errors.append(np.mean(sample_errors))
#     errors[alpha] = alpha_errors
#
# # Plot results
# plt.figure(figsize=(10, 6))
# for alpha in alpha_values:
#     plt.loglog(N_values, errors[alpha], 'o-', label=f'α={alpha}')
#
# # Add reference slopes
# for p in [1, 2, 3]:
#     ref = N_values[0]**(-p) * (N_values[0]/N_values)**p
#     plt.loglog(N_values, ref, '--', label=f'N^{-p}')
#
# plt.grid(True)
# plt.xlabel('N')
# plt.ylabel('Error')
# plt.title('Spectral Differentiation Error vs Grid Size')
# plt.legend()
# plt.show()
