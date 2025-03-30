import numpy as np

# Load data from text file
data = np.loadtxt('importance_sample.txt', dtype=float)

# Parameters
n_bootstrap = 10000  # Number of bootstrap samples
n_samples = len(data)  # Size of each bootstrap sample (same as original dataset)

# Perform bootstrap resampling
bootstrap_means = []
for _ in range(n_bootstrap):
    sample = np.random.choice(data, size=n_samples, replace=True)  # Resample with replacement
    bootstrap_means.append(np.mean(sample))

# Convert to NumPy array for convenience
bootstrap_means = np.array(bootstrap_means)

# Calculate the 90% confidence lower bound (5th percentile)
lower_bound = np.percentile(bootstrap_means, 10)

print(f"90% Confidence Lower Bound of the Mean: {lower_bound}")

# Plot the bootstrap distribution
import matplotlib.pyplot as plt
plt.hist(bootstrap_means, bins=30, density=True, alpha=0.7, edgecolor='k')
plt.axvline(lower_bound, color='red', linestyle='--', label=f'90% Lower Bound: {lower_bound:.2f}')
plt.xlabel('Bootstrap Mean')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of the Mean')
plt.legend()
plt.show()
