import numpy as np
import matplotlib.pyplot as plt

def Normalization(return_list,up_b,low_b):
    # """Calculate the importance weight and importance-weighted return for a trajectory."""
    # for i in range(len(return_list)):
    #     return_list = (return_list[i]-low_b)/(up_b-low_b)
    return [(x - low_b) / (up_b - low_b) for x in return_list]

return_list = np.loadtxt("./Trajectories/0.1_0.9/return_list.txt")
up_b= max(return_list)
low_b = min(return_list)
normalized = Normalization(return_list,up_b,low_b)
print(up_b,low_b)


# # Parameters
# n_bootstrap = 10000  # Number of bootstrap samples
# n_samples = len(normalized)  # Size of each bootstrap sample (same as original dataset)

# # Perform bootstrap resampling
# bootstrap_means = []
# for _ in range(n_bootstrap):
#     sample = np.random.choice(normalized, size=n_samples, replace=True)  # Resample with replacement
#     bootstrap_means.append(np.mean(sample))

# # Convert to NumPy array for convenience
# bootstrap_means = np.array(bootstrap_means)
# np.savetxt("bootstrap_Mean.txt",bootstrap_means)
# # Calculate the 90% confidence lower bound (5th percentile)
# # lower_bound = np.percentile(normalized, 10)
# lower_bound = np.percentile(bootstrap_means, 10)

# print(f"90% Confidence Lower Bound of the Mean: {lower_bound}")

# # Plot the bootstrap distributions
# import matplotlib.pyplot as plt

# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.figsize'] = (12, 10)
# plt.rcParams['font.size'] = 40
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["axes.linewidth"] = 2

# plt.hist(bootstrap_means, bins=30, density=True, alpha=0.7, edgecolor='k')
# plt.axvline(lower_bound, color='red', linestyle='--', label=f'90% Lower Bound: {lower_bound:.3f}')
# plt.xlabel('Bootstrap Mean', weight='bold',fontsize=40)
# plt.ylabel('Density',weight='bold',fontsize=40)
# plt.title('Bootstrap Distribution of the Mean Normalized Reward of Eval_Policy [0.1,0.9]', weight='bold',fontsize=45)
# plt.xticks(weight='bold')
# plt.yticks(weight='bold')
# plt.legend()
# plt.show()


# plt.hist(normalized, bins=30, density=True, alpha=0.7, edgecolor='k')
# plt.axvline(lower_bound, color='red', linestyle='--', label=f'90% Lower Bound Ground Truth: {lower_bound:.2f}')
# plt.xlabel('Bootstrap Mean')
# plt.ylabel('Density')
# plt.title('Distribution of Normalized Reward of Eval_Policy [0.1,0.9]')
# plt.legend()
# plt.show()



# # Plot the histogram
# plt.hist(importance_sample, bins=30, edgecolor='k', alpha=0.7)  # Adjust 'bins' as needed

# # Add labels and title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of Importance Sampled Data')

# # Show the plot
# plt.show()