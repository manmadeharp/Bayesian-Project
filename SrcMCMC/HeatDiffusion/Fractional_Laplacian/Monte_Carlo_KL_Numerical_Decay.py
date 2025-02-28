# karhunen_expansion Monte Carlo up to N = 100 
# computing what the function looks like over a range of x for each given sample
# karhunen_expansion Monte Carlo up to N = 100 
# computing what the function looks like over a range of x for each given sample


import numpy as np
import matplotlib.pyplot as plt
import scipy

rg = np.random.RandomState()

def karhunen_expansion(x, k, alpha):
    # Precompute k * np.pi once and use it for eigenvalue and eigenfunction computation
    k_pi = np.arange(1, k + 1) * np.pi
    
    # Vectorized computation of eigenvalues (k * pi) ^ alpha
    eigenvalue = np.power(k_pi, -alpha)
    
    # Compute the eigenfunctions sin(k * pi * x) for each x
    # We need to expand k_pi and x to align their dimensions for element-wise multiplication
    eigenfunction = np.sin(k_pi[:, None] * x)  # Shape (k, len(x))
    
    # Generate k normal random variables
    rn_k = rg.normal(0, 1, k)
    
    # Compute the Karhunen-Loève expansion for each x
    return np.sum(eigenvalue[:, None] * eigenfunction * rn_k[:, None], axis=0)

# # Testing the function with k=100 and x=10 values
# k = 100000
# x = np.linspace(0, 1, 10000)
#
# #print(karhunen_expansion(x, k, 1))
# alpha = 1
# y = karhunen_expansion(x, k, alpha)
#
# # Plotting the results
# plt.plot(x, y, label=f'Karhunen Expansion (k={k}, alpha={alpha})')
# plt.title('Karhunen-Loève Expansion')
# plt.xlabel('x')
# plt.ylabel('Expansion Value')
# plt.grid(True)
# plt.legend()
# plt.show()



k_values = [100000, 200000, 300000]  # Different k values to explore
alpha_values = np.linspace(1, 8, 8)  # 8 different alpha values between 0.1 and 2

x = np.linspace(0, 10, 1000)  # Range for x values (more points for smoothness)

# Loop over each alpha value and generate individual plot windows
for j, alpha in enumerate(alpha_values):
    plt.figure()  # Create a new figure window for each alpha
    # Plot the Karhunen-Loève expansion for all k values for this alpha
    for k in range(1, 4):
        y = karhunen_expansion(x, k, alpha)
        plt.plot(x, y, label=f'k={k}')
    
    # Set title, labels, and grid for the current plot
    plt.title(f'Karhunen-Loève Expansion (α={alpha:.2f})')
    plt.xlabel('x')
    plt.ylabel('Expansion Value')
    plt.grid(True)
    plt.legend()

    # Show the current figure
    plt.show()  # Display the current plot window
