import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import pandas as pd

def simulate_delay(n, sigma, delta, k,
                   trials=20, T=100, window=10,
                   t_star=50, alpha=0.01):
    """
    Simulate the average detection delay for a dynamic DAG changing at time t_star.
    - n: number of nodes
    - sigma: noise standard deviation
    - delta: magnitude of weight changes
    - k: number of edges to flip at t_star
    - trials: number of Monte Carlo trials
    - T: total time steps
    - window: sliding window size for Lasso
    - t_star: time of change
    - alpha: regularization parameter for Lasso
    """
    delays = []
    for _ in range(trials):
        # 1) Generate initial random DAG
        p_edge = 0.1
        A = (np.random.rand(n, n) < p_edge).astype(float)
        np.fill_diagonal(A, 0)
        W = A * np.random.uniform(delta, 2*delta, size=(n, n))
        
        # 2) Create weight sequence with change at t_star
        W_seq = [W.copy() for _ in range(T+1)]
        # Flip k random edges at t_star
        possible = [(i, j) for i in range(n) for j in range(n) if i != j]
        flips = np.random.choice(len(possible), k, replace=False)
        for idx in flips:
            i, j = possible[idx]
            W[i, j] = 0 if W[i, j] != 0 else np.random.uniform(delta, 2*delta)
        for t in range(t_star, T+1):
            W_seq[t] = W
        
        # 3) Simulate data
        X = np.zeros((T+1, n))
        X[0] = np.random.randn(n)
        for t in range(1, T+1):
            X[t] = X[t-1].dot(W_seq[t-1]) + np.random.randn(n) * sigma
        
        # 4) Online detection via sliding-window Lasso
        prev_adj = None
        detected = False
        for t in range(window, T+1):
            X_win = X[t-window:t]
            Y_win = X[t-window+1:t+1]
            # Fit Lasso per node
            W_est = np.zeros((n, n))
            for j in range(n):
                model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
                model.fit(X_win, Y_win[:, j])
                W_est[:, j] = model.coef_
            # Threshold to get adjacency
            adj = (np.abs(W_est) > (delta/2)).astype(int)
            # Check for change
            if prev_adj is not None and not detected:
                if not np.array_equal(adj, prev_adj):
                    delays.append(t - t_star)
                    detected = True
            prev_adj = adj.copy()
        # If never detected, assign maximum delay
        if not detected:
            delays.append(T - t_star)
    return np.mean(delays)

if __name__ == "__main__":
    # Example parameter sweep: vary network size n
    n_list = [10, 50, 100]
    sigma_list = [0.05, 0.1, 0.2, 0.4]   # noise std dev
    delta_list = [0.2, 0.5, 1.0, 2.0]    # edge‐weight change magnitude
    # 1) Sweep network size
    results_n = []
    for n in n_list:
        d = simulate_delay(n=n, sigma=0.1, delta=0.5, k=3)
        results_n.append({'param': 'n', 'value': n, 'delay': d})

    # 2) Sweep noise level
    results_sigma = []
    for sigma in sigma_list:
        d = simulate_delay(n=50, sigma=sigma, delta=0.5, k=3)
        results_sigma.append({'param': 'sigma', 'value': sigma, 'delay': d})

    # 3) Sweep change magnitude
    results_delta = []
    for delta in delta_list:
        d = simulate_delay(n=50, sigma=0.1, delta=delta, k=3)
        results_delta.append({'param': 'delta', 'value': delta, 'delay': d})

    df_n    = pd.DataFrame(results_n)
    df_sig  = pd.DataFrame(results_sigma)
    df_del  = pd.DataFrame(results_delta)

    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    axes[0].plot(df_n['value'],  df_n['delay'], marker='o')
    axes[0].set_title('Delay vs Network Size')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('Avg Delay')

    axes[1].plot(df_sig['value'],df_sig['delay'], marker='o')
    axes[1].set_title('Delay vs Noise σ')
    axes[1].set_xlabel('σ')

    axes[2].plot(df_del['value'],df_del['delay'], marker='o')
    axes[2].set_title('Delay vs Change Δ')
    axes[2].set_xlabel('Δ')

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    #Saving results to CSV
    df_n.to_csv('results_n.csv', index=False)
    df_sig.to_csv('results_sigma.csv', index=False)
    df_del.to_csv('results_delta.csv', index=False)
  
