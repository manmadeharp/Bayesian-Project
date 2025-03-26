import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Callable, Tuple, List, Optional
import tqdm

def compute_posterior_predictive(
    sampler, 
    target_distribution,
    x_pred: np.ndarray,
    t_pred: np.ndarray,
    n_samples: int = 1000,
    burn_in: int = 2000,
    thin: int = 10,
    add_noise: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the posterior predictive distribution for the heat equation.
    
    Args:
        sampler: PCN sampler after MCMC has been run
        target_distribution: Target distribution object containing the heat equation solver
        x_pred: Spatial points at which to evaluate the prediction
        t_pred: Time points at which to evaluate the prediction
        n_samples: Number of posterior samples to use
        burn_in: Number of initial MCMC samples to discard
        thin: Thinning factor for MCMC samples
        add_noise: Whether to add observation noise to predictions
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - mean: Mean of posterior predictive distribution
        - lower: Lower 95% credible interval
        - upper: Upper 95% credible interval
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate available samples after burn-in
    available_samples = sampler._index - burn_in
    
    # Determine which indices to use
    if available_samples <= 0:
        raise ValueError(f"Burn-in ({burn_in}) exceeds available samples ({sampler._index})")
    
    # Calculate maximum number of samples after thinning
    max_samples = available_samples // thin
    n_samples = min(n_samples, max_samples)
    
    # Sample indices
    indices = np.linspace(burn_in, sampler._index - 1, n_samples, dtype=int)
    
    # Initialize storage for predictions
    predictions = np.zeros((n_samples, len(x_pred)))
    
    # Get posterior samples and compute predictions
    print(f"Computing posterior predictive distribution with {n_samples} samples...")
    for i, idx in enumerate(tqdm.tqdm(indices)):
        # Get function from MCMC sample
        func = sampler.get_function(idx)
        
        # Solve heat equation for this function
        prediction = target_distribution._solve_heat_equation_impl(func, x_pred, t_pred)
        
        # Add noise if requested
        if add_noise:
            prediction += target_distribution.noise_level * np.random.randn(len(x_pred))
        
        predictions[i] = prediction
    
    # Compute statistics
    mean = np.mean(predictions, axis=0)
    lower = np.percentile(predictions, 2.5, axis=0)
    upper = np.percentile(predictions, 97.5, axis=0)
    
    return mean, lower, upper

def plot_posterior_predictive(
    x_pred: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    true_data: Optional[np.ndarray] = None,
    observed_x: Optional[np.ndarray] = None,
    observed_y: Optional[np.ndarray] = None,
    title: str = "Posterior Predictive Distribution"
) -> None:
    """
    Plot the posterior predictive distribution with credible intervals.
    
    Args:
        x_pred: Spatial points at which predictions were evaluated
        mean: Mean of posterior predictive distribution
        lower: Lower credible interval
        upper: Upper credible interval
        true_data: True data (if available)
        observed_x: Observed x values (if available)
        observed_y: Observed y values (if available)
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the credible interval
    plt.fill_between(x_pred, lower, upper, color='lightblue', alpha=0.5, label='95% Credible Interval')
    
    # Plot the posterior mean
    plt.plot(x_pred, mean, 'b-', linewidth=2, label='Posterior Mean')
    
    # Plot true data if available
    if true_data is not None:
        plt.plot(x_pred, true_data, 'g--', linewidth=2, label='True Solution')
    
    # Plot observed data if available
    if observed_x is not None and observed_y is not None:
        plt.scatter(observed_x, observed_y, color='red', s=20, alpha=0.6, label='Observations')
    
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)

def posterior_predictive_analysis(
    sampler, 
    target_distribution,
    domain_length: float = 1.0,
    n_grid_points: int = 300,
    prediction_times: List[float] = [0.01, 0.05, 0.1, 0.2],
    n_samples: int = 1000,
    burn_in: int = 2000
) -> None:
    """
    Comprehensive posterior predictive analysis at multiple time points.
    
    Args:
        sampler: PCN sampler after MCMC has been run
        target_distribution: Target distribution object containing the heat equation solver
        domain_length: Length of spatial domain
        n_grid_points: Number of grid points for prediction
        prediction_times: List of times at which to evaluate predictions
        n_samples: Number of posterior samples to use
        burn_in: Number of initial MCMC samples to discard
    """
    # Create prediction grid
    x_pred = np.linspace(0, domain_length, n_grid_points)
    
    # Compute true solution for reference at each time point
    true_solutions = {}
    for t in prediction_times:
        true_solutions[t] = target_distribution._solve_heat_equation_impl(
            target_distribution.true_initial, x_pred, t
        )
    
    # Create a figure for all time points
    plt.figure(figsize=(15, 10))
    n_rows = (len(prediction_times) + 1) // 2
    n_cols = min(2, len(prediction_times))
    
    for i, t in enumerate(prediction_times):
        print(f"Processing time t = {t}")
        
        # Compute posterior predictive
        mean, lower, upper = compute_posterior_predictive(
            sampler, 
            target_distribution,
            x_pred,
            np.full(len(x_pred), t),
            n_samples=n_samples,
            burn_in=burn_in,
            add_noise=False
        )
        
        # Add subplot
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot credible interval
        plt.fill_between(x_pred, lower, upper, color='lightblue', alpha=0.5, label='95% CI')
        
        # Plot posterior mean
        plt.plot(x_pred, mean, 'b-', linewidth=2, label='Posterior Mean')
        
        # Plot true solution
        plt.plot(x_pred, true_solutions[t], 'g--', linewidth=2, label='True Solution')
        
        # If this time point is close to an observation time, plot observations
        # This is a simplification - in reality, you'd need to find the closest match
        observed_data_shown = False
        for j, obs_t in enumerate(target_distribution.t_obs[:10]):  # Check first few observations
            if abs(t - obs_t) < 0.001:  # If time points are close
                plt.scatter(
                    target_distribution.x_obs[j:j+10], 
                    target_distribution.data[j:j+10], 
                    color='red', s=20, alpha=0.8, label='Observations' if not observed_data_shown else None
                )
                observed_data_shown = True
        
        plt.title(f"Time t = {t:.3f}")
        plt.xlabel('x')
        plt.ylabel('Temperature')
        plt.grid(True, alpha=0.3)
        
        if i == 0:  # Only add legend to first subplot for clarity
            plt.legend()
    
    plt.tight_layout()
    plt.suptitle("Posterior Predictive Analysis at Different Time Points", y=1.02, fontsize=16)
    
    # Now compute and plot posterior predictive distribution in space-time
    space_time_analysis(sampler, target_distribution, domain_length, n_samples, burn_in)

def space_time_analysis(
    sampler, 
    target_distribution,
    domain_length: float = 1.0,
    n_samples: int = 100,
    burn_in: int = 2000
) -> None:
    """
    Space-time analysis of the posterior predictive distribution.
    
    Args:
        sampler: PCN sampler after MCMC has been run
        target_distribution: Target distribution object
        domain_length: Length of spatial domain
        n_samples: Number of posterior samples to use
        burn_in: Number of initial MCMC samples to discard
    """
    # Create space-time grid
    nx, nt = 100, 50
    x_grid = np.linspace(0, domain_length, nx)
    t_grid = np.linspace(0.001, 0.2, nt)  # Avoid t=0 for numerical stability
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Initialize arrays for mean and variance
    mean_field = np.zeros((nt, nx))
    var_field = np.zeros((nt, nx))
    
    # Compute true solution for reference
    true_field = np.zeros((nt, nx))
    for i, t in enumerate(t_grid):
        true_field[i, :] = target_distribution._solve_heat_equation_impl(
            target_distribution.true_initial, x_grid, t
        )
    
    # Sample indices for posterior samples
    available_samples = sampler._index - burn_in
    indices = np.linspace(burn_in, sampler._index - 1, n_samples, dtype=int)
    
    # Compute posterior predictive in space-time
    print("Computing space-time posterior predictive distribution...")
    all_predictions = np.zeros((n_samples, nt, nx))
    
    for s, idx in enumerate(tqdm.tqdm(indices)):
        # Get function from MCMC sample
        func = sampler.get_function(idx)
        
        # Solve heat equation for all time points
        for i, t in enumerate(t_grid):
            all_predictions[s, i, :] = target_distribution._solve_heat_equation_impl(
                func, x_grid, t
            )
    
    # Compute statistics
    mean_field = np.mean(all_predictions, axis=0)
    var_field = np.var(all_predictions, axis=0)
    
    # Create plots
    plt.figure(figsize=(18, 6))
    
    # Plot true solution
    plt.subplot(1, 3, 1)
    c1 = plt.contourf(X, T, true_field, 50, cmap='viridis')
    plt.colorbar(c1, label='Temperature')
    plt.title('True Solution')
    plt.xlabel('x')
    plt.ylabel('t')
    
    # Plot posterior mean
    plt.subplot(1, 3, 2)
    c2 = plt.contourf(X, T, mean_field, 50, cmap='viridis')
    plt.colorbar(c2, label='Temperature')
    plt.title('Posterior Mean')
    plt.xlabel('x')
    plt.ylabel('t')
    
    # Plot posterior standard deviation
    plt.subplot(1, 3, 3)
    c3 = plt.contourf(X, T, np.sqrt(var_field), 50, cmap='magma')
    plt.colorbar(c3, label='Standard Deviation')
    plt.title('Posterior Uncertainty')
    plt.xlabel('x')
    plt.ylabel('t')
    
    plt.tight_layout()

def functional_pca_analysis(
    sampler, 
    target_distribution,
    n_components: int = 3,
    n_samples: int = 1000,
    burn_in: int = 2000,
    domain_length: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Functional Principal Component Analysis on the posterior samples.
    
    Args:
        sampler: PCN sampler after MCMC has been run
        target_distribution: Target distribution object
        n_components: Number of principal components to compute
        n_samples: Number of posterior samples to use
        burn_in: Number of initial MCMC samples to discard
        domain_length: Length of spatial domain
        
    Returns:
        Tuple containing:
        - eigenvalues: Eigenvalues of the empirical covariance operator
        - eigenfunctions: Eigenfunctions of the empirical covariance operator
        - scores: Principal component scores for each sample
    """
    # Create fine grid for function evaluation
    x_grid = np.linspace(0, domain_length, 300)
    
    # Sample indices for posterior
    available_samples = sampler._index - burn_in
    indices = np.linspace(burn_in, sampler._index - 1, n_samples, dtype=int)
    
    # Initialize array for function values
    func_values = np.zeros((n_samples, len(x_grid)))
    
    # Evaluate functions on grid
    print("Evaluating posterior samples for FPCA...")
    for i, idx in enumerate(tqdm.tqdm(indices)):
        func = sampler.get_function(idx)
        func_values[i, :] = func(x_grid)
    
    # Compute posterior mean
    posterior_mean = np.mean(func_values, axis=0)
    
    # Center the functions
    centered_funcs = func_values - posterior_mean
    
    # Compute empirical covariance matrix
    cov_matrix = np.cov(centered_funcs.T)
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order of eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Retain only n_components
    eigenvalues = eigenvalues[:n_components]
    eigenfunctions = eigenvectors[:, :n_components]
    
    # Compute scores for each sample
    scores = centered_funcs @ eigenfunctions
    
    # Plot results
    plt.figure(figsize=(18, 12))
    
    # Plot eigenvalue spectrum
    plt.subplot(2, 2, 1)
    plt.bar(range(1, n_components + 1), eigenvalues)
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Spectrum')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot eigenfunctions
    plt.subplot(2, 2, 2)
    for i in range(min(3, n_components)):
        plt.plot(x_grid, eigenfunctions[:, i], label=f'PC {i+1}')
    plt.xlabel('x')
    plt.ylabel('Eigenfunction Value')
    plt.title('First 3 Eigenfunctions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot posterior mean function
    plt.subplot(2, 2, 3)
    plt.plot(x_grid, posterior_mean, 'b-', linewidth=2)
    plt.plot(x_grid, target_distribution.true_initial(x_grid), 'g--', linewidth=2, 
             label='True Function')
    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.title('Posterior Mean Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot scores distribution for first two PCs
    if n_components >= 2:
        plt.subplot(2, 2, 4)
        plt.scatter(scores[:, 0], scores[:, 1], alpha=0.5)
        plt.xlabel('PC 1 Score')
        plt.ylabel('PC 2 Score')
        plt.title('PC Scores Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add a few reconstructed functions using PC scores
        for i in range(min(5, n_samples)):
            pc_contrib = np.zeros_like(posterior_mean)
            for j in range(min(2, n_components)):
                pc_contrib += scores[i, j] * eigenfunctions[:, j]
            
            # Plot a few reconstructed functions using first 2 PCs
            if i < 2:
                plt.subplot(2, 2, 3)
                plt.plot(x_grid, posterior_mean + pc_contrib, 'r-', alpha=0.3,
                         label='Reconstruction' if i == 0 else None)
    
    plt.tight_layout()
    
    return eigenvalues, eigenfunctions, scores


