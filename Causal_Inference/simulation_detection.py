import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso 
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

def simulate_single_trial_delay(n, sigma, delta, k, T, window, t_star, alpha, seed=None):
    """
    Simulates a single trial to find the detection delay.
    This function is designed to be called in parallel by joblib.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Generate initial random DAG
    p_edge = 0.1
    A = (np.random.rand(n, n) < p_edge).astype(float)
    np.fill_diagonal(A, 0)
    W = A * np.random.uniform(delta, 2 * delta, size=(n, n))

    # 2) Create weight sequence with change at t_star
    W_seq = [W.copy() for _ in range(T + 1)]
    W_changed = W.copy()

    possible_flips = []
    for r_idx in range(n):
        for c_idx in range(n):
            if r_idx != c_idx:
                possible_flips.append((r_idx, c_idx))
    
    k_actual = min(k, len(possible_flips)) 

    if k_actual > 0:
        flip_indices = np.random.choice(len(possible_flips), k_actual, replace=False)
        for idx in flip_indices:
            i, j = possible_flips[idx]
            if W_changed[i, j] != 0:  
                W_changed[i, j] = 0
            else:  
                W_changed[i, j] = np.random.uniform(delta, 2 * delta)
    
    for t_idx in range(t_star, T + 1):
        W_seq[t_idx] = W_changed.copy()

    # 3) Simulate data
    X = np.zeros((T + 1, n))
    X[0] = np.random.randn(n) * sigma
    for t_idx in range(1, T + 1):
        X[t_idx] = X[t_idx - 1].dot(W_seq[t_idx]) + np.random.randn(n) * sigma

    # 4) Online detection via sliding-window MultiTaskLasso
    prev_adj = None
    
    # Initialize MultiTaskLasso model
    # Reduced max_iter and increased tol for speed. Tune as needed.
    mt_lasso = MultiTaskLasso(alpha=alpha, fit_intercept=False, max_iter=200, tol=1e-2) 

    for t_detect in range(window, T): # Loop up to T-1 for X_win, Y_win
        X_win = X[t_detect - window : t_detect]
        Y_win = X[t_detect - window + 1 : t_detect + 1]
        
        if X_win.shape[0] < 1 or Y_win.shape[0] < 1 or X_win.shape[0] != Y_win.shape[0]:
            if prev_adj is None:
                 prev_adj = np.zeros((n,n), dtype=int) 
            continue 

        try:
            mt_lasso.fit(X_win, Y_win)
            W_est = mt_lasso.coef_.T  
                                     
            adj = (np.abs(W_est) > (delta / 2)).astype(int)
        except Exception as e:
            if prev_adj is None:
                 adj = np.zeros((n,n), dtype=int)
            else:
                adj = prev_adj.copy() 

        if prev_adj is not None and t_detect >= t_star:
            if not np.array_equal(adj, prev_adj):
                return t_detect - t_star  # Detected
        
        prev_adj = adj.copy()
    
    return T - t_star  # Not detected or detected after T

def simulate_delay_parallel(n, sigma, delta, k,
                           trials=20, T=100, window=10,
                           t_star=50, alpha=0.01, n_cores=-1):
    """
    Simulate the average detection delay using parallel trials.
    - n_cores: Number of CPU cores to use. -1 means all available.
    """
    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()
    
    # Generate unique seeds for each trial for reproducibility and independent randomness
    seeds = np.random.randint(0, 2**32 - 1, trials)

    trial_delays = Parallel(n_jobs=n_cores)(
        delayed(simulate_single_trial_delay)(n, sigma, delta, k, T, window, t_star, alpha, seed=seeds[i])
        for i in range(trials)
    )
    return np.mean(trial_delays) if trial_delays else (T - t_star)


if __name__ == "__main__":
    # parameter sweep values
    n_list = [10, 20, 30, 40, 50, 100]       # Network size
    sigma_list = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0] # Noise std dev
    delta_list = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]  # Edge-weight change magnitude
    
    # Common simulation parameters
    k_edges_flip = 3      # Number of edges to flip
    trials_count = 20     # Number of Monte Carlo trials (increase for smoother results)
    T_time_steps = 100    # Total time steps
    window_size = 10      # Sliding window size
    t_star_change = 50    # Time of change
    alpha_lasso = 0.01    # Regularization parameter for Lasso
    num_parallel_cores = -1 # Use all available cores

    print(f"Starting simulations with {trials_count} trials per setting...")
    print(f"Using up to {multiprocessing.cpu_count() if num_parallel_cores == -1 else num_parallel_cores} cores for parallel execution.")

    # 1) Sweep network size
    results_n = []
    print("\nSweeping network size (n)...")
    for n_val in n_list:
        print(f"  Simulating for n = {n_val}")
        d = simulate_delay_parallel(n=n_val, sigma=0.1, delta=0.5, k=k_edges_flip,
                                   trials=trials_count, T=T_time_steps, window=window_size,
                                   t_star=t_star_change, alpha=alpha_lasso, n_cores=num_parallel_cores)
        results_n.append({'param': 'n', 'value': n_val, 'delay': d})

    # 2) Sweep noise level
    results_sigma = []
    print("\nSweeping noise level (sigma)...")
    for sigma_val in sigma_list:
        print(f"  Simulating for sigma = {sigma_val}")
        d = simulate_delay_parallel(n=20, sigma=sigma_val, delta=0.5, k=k_edges_flip, # n fixed
                                   trials=trials_count, T=T_time_steps, window=window_size,
                                   t_star=t_star_change, alpha=alpha_lasso, n_cores=num_parallel_cores)
        results_sigma.append({'param': 'sigma', 'value': sigma_val, 'delay': d})

    # 3) Sweep change magnitude
    results_delta = []
    print("\nSweeping change magnitude (delta)...")
    for delta_val in delta_list:
        print(f"  Simulating for delta = {delta_val}")
        d = simulate_delay_parallel(n=20, sigma=0.1, delta=delta_val, k=k_edges_flip, # n fixed
                                   trials=trials_count, T=T_time_steps, window=window_size,
                                   t_star=t_star_change, alpha=alpha_lasso, n_cores=num_parallel_cores)
        results_delta.append({'param': 'delta', 'value': delta_val, 'delay': d})

    print("\nSimulations complete. Generating plots...")

    df_n    = pd.DataFrame(results_n)
    df_sig  = pd.DataFrame(results_sigma)
    df_del  = pd.DataFrame(results_delta)

    # Plot for network size n
    plt.figure(figsize=(7, 5)) 
    plt.plot(df_n['value'], df_n['delay'], marker='o', linestyle='-')
    plt.title('Detection Delay vs Network Size (n)')
    plt.xlabel('n (Network Size)')
    plt.ylabel('Average Detection Delay')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("delay_vs_network_size.png", dpi=300)

    # Plot for noise sigma
    plt.figure(figsize=(7, 5))
    plt.plot(df_sig['value'], df_sig['delay'], marker='o', linestyle='-') 
    plt.title('Detection Delay vs Noise σ')
    plt.xlabel('σ (Noise Standard Deviation)')
    plt.ylabel('Average Detection Delay')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("delay_vs_noise_sigma.png", dpi=300)

    # Plot for change magnitude delta
    plt.figure(figsize=(7, 5))
    plt.plot(df_del['value'], df_del['delay'], marker='o', linestyle='-') 
    plt.title('Detection Delay vs Change Magnitude Δ')
    plt.xlabel('Δ (Change Magnitude)')
    plt.ylabel('Average Detection Delay')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("delay_vs_change_delta.png", dpi=300)

    plt.show()
    print("\nAll plots generated and saved.")

    #Saving results to CSV
    df_n.to_csv('results_n.csv', index=False)
    df_sig.to_csv('results_sigma.csv', index=False)
    df_del.to_csv('results_delta.csv', index=False)
