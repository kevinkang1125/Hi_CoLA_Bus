import numpy as np
import scipy.stats as stats
import tqdm
import os
import matplotlib.pyplot as plt
base_return_list = np.loadtxt("./return_list.txt")

# Normalize returns using baseline min and max
def normalize(returns, up_b, low_b):
    return (returns - low_b) / (up_b - low_b)

up_b = np.max(base_return_list)
low_b = np.min(base_return_list)

return_list = np.loadtxt("./return_list_optimized.txt")
sample = normalize(return_list, up_b, low_b)
print(np.mean(sample))

def Normalization(return_list,up_b,low_b):
    return [(x - low_b) / (up_b - low_b) for x in return_list]

return_list = np.loadtxt("./Trajectories/0.1_0.9/return_list.txt")
up_b= max(return_list)
low_b = min(return_list)
normalized = Normalization(return_list,up_b,low_b)
print(up_b,low_b)


# Parameters
n_bootstrap = 10000  # Number of bootstrap samples
n_samples = 5000  # Size of each bootstrap sample (same as original dataset)

# Perform bootstrap resampling
bootstrap_means = []
for _ in range(n_bootstrap):
    sample = np.random.choice(normalized, size=n_samples, replace=True)  # Resample with replacement
    bootstrap_means.append(np.mean(sample))

# Convert to NumPy array for convenience
bootstrap_means = np.array(bootstrap_means)

# Plot the bootstrap distributions


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 40
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

plt.hist(bootstrap_means, bins=30, density=True, alpha=0.7, edgecolor='k')
# plt.axvline(lower_bound, color='red', linestyle='--', label=f'90% Lower Bound: {lower_bound:.3f}')
plt.xlabel('Bootstrap Mean', weight='bold',fontsize=40)
plt.ylabel('Density',weight='bold',fontsize=40)
plt.title('Distribution of Normalized Reward of Optimizedpolicy', weight='bold',fontsize=45)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.legend()
plt.show()



