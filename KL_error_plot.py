import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# kl_div = np.loadtxt("./Results/KL_Trend_Check_EXP/kl_diver.txt")

def mape(y_true, y_pred):
    # Convert to numpy if inputs are PyTorch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero
    epsilon = 1e-8
    mask = np.abs(y_true) > epsilon
    return np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

data = []
for i in range(5):
    bin = i*0.1
    # folder_name = f"./Results/KL_Critical_Value_Check_EXP/{bin:.1f}-{bin+0.1:.1f}/"
    Thomas_lb = np.loadtxt(f"./Results/KL_Critical_Value_Check_EXP/{bin:.1f}-{bin+0.1:.1f}/CL_list.txt")
    # Mean = np.loadtxt("./Results/KL_Trend_Check_EXP/GT.txt")
    GT = np.loadtxt(f"./Results/KL_Critical_Value_Check_EXP/{bin:.1f}-{bin+0.1:.1f}/GT.txt")
    list = mape(Thomas_lb[:100], GT[:100])
    data.append(list)



# # Example: several lists of numerical data
# data1 = [0.2, 0.3, 0.25, 0.4, 0.35]
# data2 = [0.5, 0.55, 0.52, 0.6, 0.58]
# data3 = [0.1, 0.15, 0.12, 0.18, 0.2]

# # Combine into one list of lists
# data = [data1, data2, data3]

# Boxplot
# plt.figure(figsize=(6, 5))
# plt.boxplot(data, patch_artist=True)
# plt.xticks([1, 2, 3, 4, 5], ['0.0-0.1', '0.1-0.2', '0.2-0.3','0.3-0.4','0.4-0.5'])  # optional custom labels
# plt.ylabel("Value")
# plt.title("Absolute Percentage Error")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# Define a list of colors (1 per box)
color = 'skyblue'

# Create the boxplot
fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(data, patch_artist=True)

# Color each box
for patch in bp['boxes']:
    patch.set_facecolor(color)

# Optional: Customize medians
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)

# Customize ticks
ax.set_xticklabels(['0.0-0.1', '0.1-0.2', '0.2-0.3','0.3-0.4','0.4-0.5'],fontsize=19)
ax.tick_params(axis='y', labelsize=19)
ax.set_ylabel("Absolute Percentage Error",fontsize=20)
ax.set_xlabel("KL Divergence",fontsize=20)
# ax.set_title("Absolute Percentage Error")
plt.grid(True)
plt.tight_layout()
plt.show()