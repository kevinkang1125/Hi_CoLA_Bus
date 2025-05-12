import numpy as np
import scipy.stats as stats
import tqdm
import os
base_return_list = np.loadtxt("./return_list.txt")

# Normalize returns using baseline min and max
def normalize(returns, up_b, low_b):
    return (returns - low_b) / (up_b - low_b)

up_b = np.max(base_return_list)
low_b = np.min(base_return_list)
m = 4
# for m in range(2):
bin = m*0.1
folder_name = f"./Results/KL_Critical_Value_Check_EXP/{bin:.1f}-{bin+0.1:.1f}/"
GT_lb_ls = []
for i in tqdm.tqdm(range(100)):
    file_name = f"return_list_{i}.txt"
    return_list = np.loadtxt(os.path.join(folder_name, file_name))
    sample = normalize(return_list, up_b, low_b)
    # Sample statistics
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=0)  # use ddof=1 for sample standard deviation
    n = len(sample)
    # 95% confidence level
    alpha = 0.2
    z_score = stats.norm.ppf(1 - alpha/2)  
    # Margin of error
    margin_error = z_score * (sample_std / np.sqrt(n))
    # Confidence interval
    ci_lower = sample_mean - margin_error
    GT_lb_ls.append(ci_lower)
np.savetxt(f"./Results/KL_Critical_Value_Check_EXP/{bin:.1f}-{bin+0.1:.1f}/GT.txt", GT_lb_ls)