# Re-import everything after execution state reset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Re-load return data
return_list = np.loadtxt("./Test/return_list_3.txt")
base_return_list = np.loadtxt("./return_list.txt")

# Normalize returns using baseline min and max
def normalize(returns, up_b, low_b):
    return (returns - low_b) / (up_b - low_b)

up_b = np.max(base_return_list)
low_b = np.min(base_return_list)

# Bootstrap with convergence detection
def bootstrap(return_list, up_b, low_b, batch_size=2000,num_bootstrap=200, var_tol=1e-4, window=20):

    max_size = len(return_list)

    sizes = []
    lower_bounds = []

    for n in tqdm(range(batch_size, max_size + 1, 1000), desc="Bootstrapping"):
        current_bounds = []

        for _ in range(num_bootstrap):
            sample = np.random.choice(return_list, size=n, replace=True)
            norm_sample = normalize(sample, up_b, low_b)
            bound = np.percentile(norm_sample, 10)
            current_bounds.append(bound)

        avg_bound = np.mean(current_bounds)
        lower_bounds.append(avg_bound)
        sizes.append(n)

        # if len(lower_bounds) >= window:
        #     recent = lower_bounds[-window:]
        #     if np.var(recent) < var_tol*np.mean(recent):
        #         print(f"✅ Bootstrapped lower bound converged at sample size ≈ {n}")
        #         break

    return sizes, lower_bounds

lb_ls = []
epoch_ls = []
# for i in tqdm(range(100)):
# Run the bootstrapped convergence search
i = 2
# return_list = np.loadtxt(f"./Test/return_list_{3}.txt")
sizes_boot, bounds_boot = bootstrap(return_list, up_b, low_b)
# lb_ls.append(bounds_boot[-1])
# epoch_ls.append(sizes_boot[-1])

# np.savetxt("./Test/BS_epoch_list.txt", epoch_ls)
# np.savetxt("./Test/BS_lower_bound_list.txt", lb_ls)
    



# Plotting
plt.plot(sizes_boot, bounds_boot, marker='o', label='Bootstrapped 10% Lower Bound')
plt.xlabel("Sample Size")
plt.ylabel("Estimated Lower Bound (Normalized)")
plt.title("Bootstrap Convergence of 10% Lower Bound")
plt.grid(True)
plt.legend()
plt.show()
