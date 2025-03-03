# karhunen_expansion Monte Carlo up to N = 100 
# computing what the function looks like over a range of x for each given sample
# karhunen_expansion Monte Carlo up to N = 100 
# computing what the function looks like over a range of x for each given sample


import numpy as np
import matplotlib.pyplot as plt
import scipy

rg = np.random.RandomState()

def karhunen_expansion(x, k_start, k_end, alpha):
    # Precompute k * np.pi once and use it for eigenvalue and eigenfunction computation
    k_pi = np.arange(k_start, k_end) * np.pi
    
    # Vectorized computation of eigenvalues (k * pi) ^ alpha
    eigenvalue = np.power(k_pi, -alpha)
    
    # Compute the eigenfunctions sin(k * pi * x) for each x
    # We need to expand k_pi and x to align their dimensions for element-wise multiplication
    eigenfunction = np.sin(k_pi[:, None] * x)  # Shape (k, len(x))
    
    # Generate k normal random variables
    rn_k = rg.normal(0, 1, k_end - k_start)
    
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
alpha_values = np.linspace(1.4, 1.4, 1)  # 8 different alpha values between 0.1 and 2

x = np.linspace(0, 1, 1000)  # Range for x values (more points for higher resolution)

# Loop over each alpha value and generate individual plot windows
for j, alpha in enumerate(alpha_values):
    plt.figure()  # Create a new figure window for each alpha
    # Plot the Karhunen-Loève expansion for all k values for this alpha
    y = karhunen_expansion(x, 1, 100000, alpha)
    plt.plot(x, y, label='k=100000')
    for k in range(1, 8):
        y += karhunen_expansion(x, k*100000, (k+1)*100000, alpha)
        plt.plot(x, y, label=f'k={k*100000}')
    
    # Set title, labels, and grid for the current plot
    plt.title(f'Karhunen-Loève Expansion (α={alpha:.2f})')
    plt.xlabel('x')
    plt.ylabel('Expansion Value')
    plt.grid(True)
    plt.legend()

    # Show the current figure
    plt.show()  # Display the current plot window
