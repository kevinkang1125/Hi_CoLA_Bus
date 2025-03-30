import json
import numpy as np
import matplotlib.pyplot as plt


def load_json(file_path):
    """Load the JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def importance_sampling(trajectory, evaluation_policy,base_policy,up_b,low_b):
    """Calculate the importance weight and importance-weighted return for a trajectory."""
    action_list = trajectory["actions"]
    importance_weight = 1
    for j in range(len(action_list)):
        importance_weight *= (evaluation_policy[0][action_list[j][0]]*evaluation_policy[1][action_list[j][1]])/(base_policy[0][action_list[j][0]]*base_policy[1][action_list[j][1]])
    return_value  = trajectory["return"]
    norm_return = (return_value-low_b)/(up_b-low_b)
    return importance_weight * norm_return


return_list = np.loadtxt("return_list.txt")
file_path = "./trajectories.json"  # Replace with your file path
data = load_json(file_path)
evaluation_policy = [[0.6,0.4],[0.4,0.6]]
base_policy = [[0.5,0.5],[0.5,0.5]]
max_return = max(return_list)
min_return = min(return_list)
importance_sample = []
for i in range(len(data)):
    traj = data[i]
    sample = importance_sampling(traj,evaluation_policy=evaluation_policy,base_policy=base_policy,up_b = max_return,low_b = min_return)
    importance_sample.append(sample)

np.savetxt("importance_sample.txt",importance_sample)


# Plot the histogram
plt.hist(importance_sample, bins=30, edgecolor='k', alpha=0.7)  # Adjust 'bins' as needed

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Importance Sampled Data')

# Show the plot
plt.show()