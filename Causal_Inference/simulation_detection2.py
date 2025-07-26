import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import warnings


warnings.filterwarnings('ignore', category=UserWarning)

def generate_var_data(n, sigma, delta, k, p_edge, T, t_star, seed=None):
    if seed is not None:
        np.random.seed(seed)

    A = (np.random.rand(n, n) < p_edge).astype(float)
    np.fill_diagonal(A, 0)
    W_initial = A * np.random.uniform(delta, 2 * delta, size=(n, n))
    W_changed = W_initial.copy()
    possible_flips = [(r, c) for r in range(n) for c in range(n) if r != c]
    k_actual = min(k, len(possible_flips))
    if k_actual > 0:
        flip_indices_indices = np.random.choice(len(possible_flips), k_actual, replace=False)
        flip_indices = [possible_flips[i] for i in flip_indices_indices]
        for i, j in flip_indices:
            if W_changed[i, j] != 0: W_changed[i, j] = 0
            else: W_changed[i, j] = np.random.uniform(delta, 2 * delta)

    W_seq = [W_initial.copy() if t < t_star else W_changed.copy() for t in range(T + 1)]

    X = np.zeros((T + 1, n))
    X[0] = np.random.randn(n) * sigma
    for t in range(1, T + 1):
        X[t] = X[t-1] @ W_seq[t] + np.random.randn(n) * sigma

    return X

def dynamic_change_detection(X, n, T, window_size, t_star, delta, lambda_lasso, lasso_max_iter, lasso_tol, sigma):
    mt_lasso = MultiTaskLasso(alpha=lambda_lasso, fit_intercept=False, max_iter=lasso_max_iter, tol=lasso_tol)
    theta_thresh = 0.1
    X_init, Y_init = X[0:window_size], X[1:window_size+1]
    try:
        mt_lasso.fit(X_init, Y_init)
        prev_adj = (np.abs(mt_lasso.coef_.T) > theta_thresh).astype(int)
    except:
        prev_adj = np.zeros((n, n), dtype=int)

    for t_detect in range(window_size + 1, T):
        X_win, Y_win = X[t_detect-window_size:t_detect], X[t_detect-window_size+1:t_detect+1]
        try:
            mt_lasso.fit(X_win, Y_win)
            adj_curr = (np.abs(mt_lasso.coef_.T) > theta_thresh).astype(int)
        except:
            adj_curr = prev_adj.copy()

        if t_detect >= t_star and not np.array_equal(adj_curr, prev_adj):
            return t_detect - t_star
        prev_adj = adj_curr.copy()

    return T - t_star

def static_cusum_baseline(X, n, T, t_star, sigma, lambda_lasso, lasso_max_iter, lasso_tol):
    mt_lasso = MultiTaskLasso(alpha=lambda_lasso, fit_intercept=False, max_iter=lasso_max_iter, tol=lasso_tol)
    X_train, Y_train = X[0:t_star-1], X[1:t_star]
    try:
        mt_lasso.fit(X_train, Y_train)
        W_static = mt_lasso.coef_.T
    except:
        W_static = np.zeros((n, n))

    S = 0.0
    train_residuals = Y_train - X_train @ W_static
    mean_sq_resid_h0 = np.mean(np.sum(train_residuals**2, axis=1))
    c = mean_sq_resid_h0 * 1.1 
    
    eta = 5 * c 
   

    for t_monitor in range(t_star, T):
        x_prev, x_curr = X[t_monitor-1], X[t_monitor]
        x_pred = x_prev @ W_static
        residual_sq_norm = np.sum((x_curr - x_pred)**2)
        
        S = max(0, S + residual_sq_norm - c)
        
        if S > eta:
            return t_monitor - t_star # Change detected
            
    return T - t_star


def run_single_trial_delay(sim_params, seed, algorithm='dynamic'): 
    datagen_params = {
        'n': sim_params['n'], 'sigma': sim_params['sigma'], 'delta': sim_params['delta'],
        'k': sim_params['k'], 'p_edge': sim_params['p_edge'], 'T': sim_params['T'],
        't_star': sim_params['t_star']
    }
    X = generate_var_data(**datagen_params, seed=seed)
    
    if algorithm == 'dynamic':
        detection_params = {
            'n': sim_params['n'], 'T': sim_params['T'], 'window_size': sim_params['window_size'], 
            't_star': sim_params['t_star'], 'delta': sim_params['delta'], 
            'lambda_lasso': sim_params['lambda_lasso'], 'lasso_max_iter': sim_params['lasso_max_iter'], 
            'lasso_tol': sim_params['lasso_tol'],
            'sigma': sim_params['sigma'] # <--- THE FIX IS HERE
        }
        return dynamic_change_detection(X, **detection_params)
        
    elif algorithm == 'baseline':
        baseline_params = {
            'n': sim_params['n'], 'T': sim_params['T'], 't_star': sim_params['t_star'], 
            'sigma': sim_params['sigma'], 'lambda_lasso': sim_params['lambda_lasso'], 
            'lasso_max_iter': sim_params['lasso_max_iter'], 'lasso_tol': sim_params['lasso_tol']
        }
        return static_cusum_baseline(X, **baseline_params)
    else:
        raise ValueError("Unknown algorithm type")

def run_single_trial_far(sim_params, seed, algorithm='dynamic'): 
    far_sim_params = sim_params.copy()
    far_sim_params['t_star'] = far_sim_params['T']

    datagen_params = {
        'n': far_sim_params['n'], 'sigma': far_sim_params['sigma'], 'delta': far_sim_params['delta'],
        'k': far_sim_params['k'], 'p_edge': far_sim_params['p_edge'], 'T': far_sim_params['T'],
        't_star': far_sim_params['t_star']
    }
    X = generate_var_data(**datagen_params, seed=seed)
    
    if algorithm == 'dynamic':
        detection_params = {
            'n': far_sim_params['n'], 'T': far_sim_params['T'], 'window_size': far_sim_params['window_size'], 
            't_star': far_sim_params['t_star'], 'delta': far_sim_params['delta'], 
            'lambda_lasso': far_sim_params['lambda_lasso'], 'lasso_max_iter': far_sim_params['lasso_max_iter'], 
            'lasso_tol': far_sim_params['lasso_tol'],
            'sigma': far_sim_params['sigma'] # <--- THE FIX IS HERE
        }
        delay = dynamic_change_detection(X, **detection_params)
        
    elif algorithm == 'baseline':
        baseline_params = {
            'n': far_sim_params['n'], 'T': far_sim_params['T'], 't_star': far_sim_params['t_star'], 
            'sigma': far_sim_params['sigma'], 'lambda_lasso': far_sim_params['lambda_lasso'], 
            'lasso_max_iter': far_sim_params['lasso_max_iter'], 'lasso_tol': far_sim_params['lasso_tol']
        }
        delay = static_cusum_baseline(X, **baseline_params)
    
    return 1 if delay < (far_sim_params['T'] - far_sim_params['t_star']) else 0
    
def simulate_parallel(run_func, sim_params, trials, n_cores, algorithm):
    from tqdm import tqdm

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()
    seeds = np.random.randint(0, 2**32 - 1, trials)
    results = Parallel(n_jobs=n_cores)(
        delayed(run_func)(sim_params, seed, algorithm)
        for seed in tqdm(seeds, desc=f"Running {algorithm}")
    )
    return np.mean(results)

if __name__ == "__main__":
    
    DEBUG_MODE = False 

    if DEBUG_MODE:
        print("--- RUNNING IN DEBUG MODE (FAST) ---")
        common_params = {
            'k': 3, 'p_edge': 0.1, 'T': 100, 'window_size': 10, 't_star': 50,
            'lambda_lasso': 0.05, 'lasso_max_iter': 200, 'lasso_tol': 1e-2,
            'n': 20, 'sigma': 0.1, 'delta': 0.5
        }
        trials = 10 # Drastically reduced trials
        n_list = [10, 20, 30]
        sigma_list = [0.1, 0.5, 1.0]
        delta_list = [0.2, 0.5, 1.0]
    else:
        print("--- RUNNING IN FULL SIMULATION MODE ---")
        common_params = {
            'k': 3, 'p_edge': 0.1, 'T': 300,             
            'window_size': 20, 't_star': 150,    
            'lambda_lasso': 0.01, 'lasso_max_iter': 300, 
            'lasso_tol': 1e-3,
            'n': 50, 'sigma': 0.1, 'delta': 0.5
        }
        trials = 50 
        n_list = [10, 20, 30, 40, 50, 100]
        sigma_list = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        delta_list = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    n_cores = -1
    
    print(f"Starting simulations with {trials} trials per setting...")
    print("\nEvaluating False Alarm Rate (FAR)...")
    print("Running FAR for Proposed Algorithm...")
    far_dynamic = simulate_parallel(run_single_trial_far, common_params, trials, n_cores, 'dynamic')
    print("Running FAR for Baseline Algorithm...")
    far_baseline = simulate_parallel(run_single_trial_far, common_params, trials, n_cores, 'baseline')
    print(f"  Proposed Algorithm FAR: {far_dynamic:.2%}")
    print(f"  Baseline Algorithm FAR: {far_baseline:.2%}")
    
    param_sweeps = {
        'n': ('Network Size (n)', [10, 20, 30, 40, 50, 100]),
        'sigma': ('Noise σ', [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]),
        'delta': ('Change Magnitude Δ', [0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    }

    results_df = pd.DataFrame()

    for param, (xlabel, values) in param_sweeps.items():
        print(f"\nSweeping {param}...")
        delays_dynamic = []
        delays_baseline = []
        for v in values:
            print(f"  Simulating for {param} = {v}")
            current_params = common_params.copy()
            current_params[param] = v

            delay_dyn = simulate_parallel(run_single_trial_delay, current_params, trials, n_cores, 'dynamic')
            delay_base = simulate_parallel(run_single_trial_delay, current_params, trials, n_cores, 'baseline')

            delays_dynamic.append(delay_dyn)
            delays_baseline.append(delay_base)

        # Plotting
        plt.figure(figsize=(7, 5))
        plt.plot(values, delays_dynamic, marker='o', linestyle='-', label='Proposed Algorithm')
        plt.plot(values, delays_baseline, marker='s', linestyle='--', label='Static + CUSUM Baseline')
        plt.title(f'Detection Delay vs {xlabel}')
        plt.xlabel(xlabel)
        plt.ylabel('Average Detection Delay')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"delay_vs_{param}.png", dpi=300)

        # Store results 
        temp_df = pd.DataFrame({
            'param_swept': param,
            'param_value': values,
            'delay_dynamic': delays_dynamic,
            'delay_baseline': delays_baseline
        })
        results_df = pd.concat([results_df, temp_df])

    plt.show()
    print("\nAll plots generated and saved.")

    # Saving all results to a single CSV
    results_df.to_csv('simulation_results_with_baseline.csv', index=False)

    # Save FAR results
    far_results = pd.DataFrame({
        'algorithm': ['Proposed', 'Baseline'],
        'FAR': [far_dynamic, far_baseline]
    })
    far_results.to_csv('far_results.csv', index=False)
