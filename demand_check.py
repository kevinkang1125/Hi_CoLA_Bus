import numpy as np
import pandas as pd


# def get_demand(time_period):
#     demand_dist = []
#     if time_period == 1:
#         expected_arrivals = [2,0.2,1,0.4,0.6]
#     elif time_period == 2:
#         expected_arrivals = [0.4,0.2,0.4,0.2,0.2]
#     else:
#         expected_arrivals = [1.6,0.4,1.4,0.4,0.8]
#     for i in range(len(expected_arrivals)):
#         demand_dist.append(np.random.poisson(expected_arrivals[i]))
#     # print(demand_dist)    
#     return demand_dist

# def ridership_cal(deviate_1,deviate_2,demand_dist,tol):
#     ridership = demand_dist[0] + demand_dist[1]*deviate_1 + demand_dist[2]*((1-tol)**(deviate_1)) + demand_dist[3]*deviate_2+ demand_dist[4]*((1-tol)**(deviate_2))
    
#     return ridership

# sample_rec = []
# reward = 0
# gamma = 1
# num_traj = 100000
# for traj in range(num_traj):
#     for i in range(5):
#         ridership = np.sum(get_demand(1))
#         reward  = reward*gamma + ridership

#     sample_rec.append(reward)
# np.savetxt("sample_rec.txt", sample_rec,delimiter=",")

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from multiprocessing import Pool, cpu_count
import time
from functools import partial

# Set random seed for reproducibility per process
#def init_worker(seed):
#  np.random.seed(seed)

def get_demand(time_period):
    """Generate demand distribution based on time period"""
    demand_dist = []
    if time_period == 1:
        expected_arrivals = [2, 8, 20, 9, 20]
    elif time_period == 2:
        expected_arrivals = [0.4, 0.2, 0.4, 0.2, 0.2]
    else:
        expected_arrivals = [1.6, 0.4, 1.4, 0.4, 0.8]
    
    for expected in expected_arrivals:
        demand_dist.append(np.random.poisson(expected))
    return demand_dist

def simulate_single_trajectory(traj_idx, gamma=0.99, n_steps=5, time_period=1):
    """Simulate a single trajectory and return cumulative discounted reward"""
    reward = 0
    for i in range(n_steps):
        ridership = np.sum(get_demand(time_period))
        if i == 0:
            reward = ridership
        else:
            reward = reward * gamma + ridership
    return reward

def simulate_batch(batch_size, gamma=1, n_steps=5, time_period=1):
    """Simulate a batch of trajectories"""
    rewards = []
    for _ in range(batch_size):
        reward = 0
        for i in range(n_steps):
            ridership = np.sum(get_demand(time_period))
            if i == 0:
                reward = ridership
            else:
                reward = reward * gamma + ridership
        rewards.append(reward)
    return rewards

def parallel_simulation(num_traj=100000, n_workers=None, gamma=0.99, n_steps=5, time_period=1):
    """Run simulation in parallel using multiprocessing"""
    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # Cap at 8 workers
    
    print(f"Running simulation with {n_workers} workers...")
    
    # Calculate batch sizes
    batch_size = num_traj // n_workers
    remainder = num_traj % n_workers
    batch_sizes = [batch_size] * n_workers
    for i in range(remainder):
        batch_sizes[i] += 1
    
    # Start timer
    start_time = time.time()
    
    # Create partial function with fixed parameters
    simulate_func = partial(simulate_batch, gamma=gamma, n_steps=n_steps, time_period=time_period)
    
    # Run parallel simulation
    with Pool(processes=n_workers) as pool:
        results = pool.map(simulate_func, batch_sizes)
    
    # Flatten results
    sample_rec = []
    for batch_rewards in results:
        sample_rec.extend(batch_rewards)
    
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    
    return np.array(sample_rec)

def plot_statistics(sample_rec, save_file=True):
    """Plot comprehensive statistics of the simulation results"""
    # Calculate statistics
    mean = np.mean(sample_rec)
    median = np.median(sample_rec)
    std = np.std(sample_rec)
    min_val = np.min(sample_rec)
    max_val = np.max(sample_rec)
    q1 = np.percentile(sample_rec, 25)
    q3 = np.percentile(sample_rec, 75)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Histogram with KDE
    ax1 = axes[0, 0]
    n, bins_edges, patches = ax1.hist(sample_rec, bins=50, density=True, 
                                       alpha=0.7, color='skyblue', 
                                       edgecolor='black', label='Histogram')
    # Add KDE
    kde_data = stats.gaussian_kde(sample_rec)
    x_range = np.linspace(min_val, max_val, 200)
    ax1.plot(x_range, kde_data(x_range), 'r-', linewidth=2, label='KDE')
    ax1.axvline(mean, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    ax1.axvline(median, color='orange', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Density')
    ax1.set_title('Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    box = ax2.boxplot(sample_rec, vert=True, patch_artist=True, 
                      showmeans=True, meanline=True)
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][0].set_edgecolor('navy')
    ax2.set_ylabel('Reward')
    ax2.set_title('Box Plot')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Mean: {mean:.3f}\nMedian: {median:.3f}\nStd: {std:.3f}\n'
    stats_text += f'Min: {min_val:.3f}\nMax: {max_val:.3f}\n'
    stats_text += f'Q1: {q1:.3f}\nQ3: {q3:.3f}'
    ax2.text(1.4, median, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Q-Q plot
    ax3 = axes[0, 2]
    stats.probplot(sample_rec, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Check)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Running average
    ax4 = axes[1, 0]
    window_size = 1000
    running_avg = np.convolve(sample_rec, np.ones(window_size)/window_size, mode='valid')
    ax4.plot(running_avg, linewidth=1)
    ax4.set_xlabel('Trajectory Index')
    ax4.set_ylabel('Running Average Reward')
    ax4.set_title(f'Running Average (window={window_size})')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(mean, color='red', linestyle='--', alpha=0.5, label='Overall Mean')
    ax4.legend()
    
    # 5. CDF
    ax5 = axes[1, 1]
    sorted_data = np.sort(sample_rec)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax5.plot(sorted_data, cumulative, 'b-', linewidth=2)
    ax5.set_xlabel('Reward')
    ax5.set_ylabel('Cumulative Probability')
    ax5.set_title('Cumulative Distribution Function')
    ax5.grid(True, alpha=0.3)
    
    # Add percentile lines
    for p in [25, 50, 75]:
        percentile_val = np.percentile(sample_rec, p)
        ax5.axvline(percentile_val, color='red', linestyle=':', alpha=0.5)
        ax5.text(percentile_val, 0.02, f'{p}%', fontsize=8, color='red')
    
    # 6. Convergence plot
    ax6 = axes[1, 2]
    cumulative_mean = np.cumsum(sample_rec) / np.arange(1, len(sample_rec) + 1)
    ax6.plot(cumulative_mean, linewidth=1)
    ax6.set_xlabel('Number of Trajectories')
    ax6.set_ylabel('Cumulative Mean')
    ax6.set_title('Convergence of Mean')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(mean, color='red', linestyle='--', alpha=0.5, label=f'Final Mean: {mean:.3f}')
    ax6.legend()
    
    plt.suptitle('Ridership Simulation Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Count:      {len(sample_rec)}")
    print(f"Mean:       {mean:.6f}")
    print(f"Median:     {median:.6f}")
    print(f"Std Dev:    {std:.6f}")
    print(f"Variance:   {np.var(sample_rec):.6f}")
    print(f"Min:        {min_val:.6f}")
    print(f"Max:        {max_val:.6f}")
    print(f"Range:      {max_val - min_val:.6f}")
    print(f"Q1 (25%):   {q1:.6f}")
    print(f"Q3 (75%):   {q3:.6f}")
    print(f"IQR:        {q3 - q1:.6f}")
    print(f"Skewness:   {stats.skew(sample_rec):.6f}")
    print(f"Kurtosis:   {stats.kurtosis(sample_rec):.6f}")
    print("="*50)
    
    if save_file:
        np.savetxt("sample_rec.txt", sample_rec, delimiter=",")
        print("\nResults saved to sample_rec.txt")
    
    return sample_rec



# Main execution
if __name__ == "__main__":
    # Parameters
    num_traj = 100000
    gamma = 1
    n_steps = 5
    time_period = 1
    
    # Run optimized simulation
    sample_rec = parallel_simulation(num_traj=num_traj, 
                                    n_workers=8,  # Auto-detect
                                    gamma=gamma, 
                                    n_steps=n_steps,
                                    time_period=time_period)
    
    # Plot statistics
    plot_statistics(sample_rec, save_file=True)
    
    # Optional: Compare performance (uncomment to run)
    # compare_performance()