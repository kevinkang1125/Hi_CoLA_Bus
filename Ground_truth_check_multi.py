import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def Normalization(return_list, up_b, low_b):
    return [(x - low_b) / (up_b - low_b) for x in return_list]

def population_mean(return_list, base_return_list):
    up_b = max(base_return_list)
    low_b = min(base_return_list)
    normalized = Normalization(return_list, up_b, low_b)

    return np.mean(normalized)
base_return_list = np.loadtxt("./return_list.txt")
mean_ls = []
# === Load data ===
for i in tqdm(range(100), desc="Processing Runs"):
    return_list = np.loadtxt(f"./Test/return_list_{i}.txt")
    means = population_mean(return_list, base_return_list)
    mean_ls.append(means)

np.savetxt("./Test/GT.txt", mean_ls)
