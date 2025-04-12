import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from Behaviour_pi import Behavioural, Policy
import gym
import time
import flexible_bus
from policy import model_1, model_2  
import json
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import torch
import torch.nn as nn
import pandas as pd




def importance_sample_local(traj_list,eval_pi,behave_pi,traj_len=5):
    return_list = traj_list.iloc[:, 1].to_numpy()
    up_b, low_b = max(return_list), min(return_list)
    iw_ls = []
    wr_ls=[] #weighte return list
    # up_b, low_b = max(return_list), min(return_list)
    for j in range(len(traj_list)):
        for i in range(traj_len):
            obs = traj_list.iloc[j][f"obs_{i}"]
            a = traj_list.iloc[j][f"direct_action_{i}"]
            obs = torch.tensor(obs,dtype=torch.float32)
            eval_out = eval_pi(obs)
            behave_out = behave_pi(obs)
            eval_policy = [[1-eval_out[0].item(),eval_out[0].item()],[1-eval_out[1].item(),eval_out[1].item()]]
            base_policy = [[1-behave_out[0].item(),behave_out[0].item()],[1-behave_out[1].item(),behave_out[1].item()]]
            importance_weights =((eval_policy[0][a[0]] * eval_policy[1][a[1]]) /(base_policy[0][a[0]] * base_policy[1][a[1]]))
            iw_ls.append(importance_weights)
        importance_weight = np.prod(iw_ls)
        norm_return = (traj_list.iloc[j]["return"] - low_b) / (up_b - low_b)
        wr = norm_return*importance_weight
        wr_ls.append(wr)
    return wr_ls

# def importance_sample_global(traj_list,eval_pi,behave_pi,up_b,low_b,traj_len=5):
#     iw_ls = []
#     wr_ls=[] #weighte return list
#     for j in range(len(traj_list)):
#         for i in range(traj_len):
#             obs = traj_list.iloc[j][f"obs_{i}"]
#             a = traj_list.iloc[j][f"direct_action_{i}"]
#             obs = torch.tensor(obs,dtype=torch.float32)
#             eval_out = eval_pi(obs)
#             behave_out = behave_pi(obs)
#             eval_policy = [[1-eval_out[0].item(),eval_out[0].item()],[1-eval_out[1].item(),eval_out[1].item()]]
#             base_policy = [[1-behave_out[0].item(),behave_out[0].item()],[1-behave_out[1].item(),behave_out[1].item()]]
#             importance_weights =((eval_policy[0][a[0]] * eval_policy[1][a[1]]) /(base_policy[0][a[0]] * base_policy[1][a[1]]))
#             iw_ls.append(importance_weights)
#         importance_weight = np.prod(iw_ls)
#         norm_return = (traj_list.iloc[j]["return"] - low_b) / (up_b - low_b)
#         wr = norm_return*importance_weight
#         wr_ls.append(wr)
#     return wr_ls

def importance_sample_global(traj_df, eval_pi, behave_pi, up_b, low_b, traj_len=5, device="cuda"):
    # Move models to GPU if needed
    eval_pi.to(device)
    behave_pi.to(device)
    eval_pi.eval()
    behave_pi.eval()

    all_obs = []
    all_actions = []
    return_values = traj_df["return"].to_numpy()

    # Preload and flatten obs/actions
    for _, row in traj_df.iterrows():
        for i in range(traj_len):
            all_obs.append(row[f"obs_{i}"])
            all_actions.append(row[f"direct_action_{i}"])

    # Convert to tensors on GPU
    obs_tensor = torch.tensor(all_obs, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long, device="cpu")  # keep on CPU for indexing

    with torch.no_grad():
        eval_out = eval_pi(obs_tensor)
        behave_out = behave_pi(obs_tensor)

    # Convert outputs to probability matrices: shape (N, 2, 2)
    # eval_out and behave_out are (N, 2): [p1, p2]
    p1_eval, p2_eval = eval_out[:, 0], eval_out[:, 1]
    p1_base, p2_base = behave_out[:, 0], behave_out[:, 1]

    # Build [1 - p, p] for both actions (so shape becomes (N, 2) for each)
    eval_prob_1 = torch.stack([1 - p1_eval, p1_eval], dim=1)  # (N, 2)
    eval_prob_2 = torch.stack([1 - p2_eval, p2_eval], dim=1)

    base_prob_1 = torch.stack([1 - p1_base, p1_base], dim=1)
    base_prob_2 = torch.stack([1 - p2_base, p2_base], dim=1)

    # Now select action probabilities using actions_tensor
    # actions_tensor shape: (N, 2)
    a1 = actions_tensor[:, 0]
    a2 = actions_tensor[:, 1]

    eval_action_probs = eval_prob_1[range(len(a1)), a1] * eval_prob_2[range(len(a2)), a2]
    base_action_probs = base_prob_1[range(len(a1)), a1] * base_prob_2[range(len(a2)), a2]

    # Importance weights
    iw = (eval_action_probs / base_action_probs).cpu().tolist()
    # Aggregate per-trajectory (assuming equal-length)
    traj_iws = np.array(iw).reshape(-1, traj_len)
    traj_weights = traj_iws.prod(axis=1)
    # print(np.shape(traj_weights))
    # Normalize returns
    norm_returns = (return_values - low_b) / (up_b - low_b)
    wr_list = norm_returns * traj_weights

    return wr_list.tolist()



def high_confidence_cal(sample,confidence_level = 0.9,tol = 1e-3):
    delta = 1 - confidence_level
    sample_size = len(sample) // 20
    random_sample = np.random.choice(sample, size=sample_size, replace=False)
    iw_returns = random_sample
    n = len(random_sample)
    c_min, c_max = 1, 50
    best_c, best_lower_bound = c_min, -float('inf')

    while (c_max - c_min) > tol:
        c_mid = (c_min + c_max) / 2
        truncated_returns = np.minimum(iw_returns, c_mid)

        empirical_mean = np.mean(truncated_returns)

        term_1 = empirical_mean
        term_2 = 7 * c_mid * np.log(2 / delta) / (3 * (n - 1))
        term_3 = np.sqrt(
            (2 * np.log(2 / delta) / ((n - 1) * n * (len(sample) - n))) *
            (n * np.sum((truncated_returns / c_mid) ** 2) - (np.sum(truncated_returns / c_mid)) ** 2)
        )
        lower_bound = term_1 - term_2 - term_3

        if lower_bound > best_lower_bound:
            best_lower_bound = lower_bound
            best_c = c_mid

        if lower_bound > best_lower_bound:
            c_min = c_mid
        else:
            c_max = c_mid
    m = len(sample)
    truncated_returns = np.minimum(sample, best_c)
    pairwise_sum = 2*(m**2)* np.var(truncated_returns, ddof=0)

    term_1 = np.mean(truncated_returns)
    term_2 = 7 * best_c * np.log(2 / delta) / (3 * (m - 1))
    term_3 = np.sqrt((np.log(2 / delta)) * pairwise_sum / (m - 1)) / m
    lower_bound = term_1 - term_2 - term_3
    return best_c, lower_bound

def cb_sliding_window_search(traj_list,return_list,eval_policy,base_policy,min_epoch,max_epoch,step,converge,confidence_level):
    """
    Function to search for the converged confidence lower bound
    """
    folder_path ="./Test/"
    return_list = traj_list.iloc[:, 1].to_numpy()
    max_return, min_return = max(return_list), min(return_list)
    lower_bound_list, iters = [], []
    for epochs in tqdm(range(min_epoch, max_epoch, step), desc="Processing Epochs"):
        sampled_ids = np.random.choice(len(traj_list), size=epochs, replace=False)
        sampled_df = traj_list[traj_list["trajectory_id"].isin(sampled_ids)]
        importance_samples = importance_sample_global(
            sampled_df, eval_policy, base_policy, max_return, min_return
        )
        c_star, lower_bound = high_confidence_cal(importance_samples,confidence_level)
        #lower_bound_list_ = lower_bound_list
        lower_bound_list.append(lower_bound)
        iters.append(epochs)
        if epochs >100000:
            if np.var(lower_bound_list[-20:])<converge*np.mean(lower_bound_list[-20:]):
                break
        
    # np.savetxt(f"{folder_path}/Lower_bound_list.txt", lower_bound_list)
    # np.savetxt(f"{folder_path}/Sample_size_list.txt", iters)
    # return epochs,lower_bound_list
    return iters,lower_bound_list



if __name__ == "__main__":
    # Parameters
    max_epochs = 1000000
    step = 2000
    min_epochs = 1000
    alpha = 0.05  # Adjust based on desired sensitivity
    traj_num = 1000000
    gamma = 0.99
    converge = 0.0001
    confidence_level = 0.9
    # base_policy = [0.5,0.5]
    converge_epoch_ls = []
    # for i in tqdm(range(9),desc="Outer_loop"):
    #     for j in tqdm(range(9),desc="Inter_loop"): 
    start_time = time.time()  # Record the start times
    base_policy = Behavioural()
    base_policy.load_state_dict(torch.load("./Behavioural_model_2.pth"))
    traj_list = pd.read_parquet("./Traject/trajectories_2.pq")
    return_list = np.loadtxt("./return_list.txt")
    lb_list = []
    # for i in range(100):
    #     eval_policy = torch.load(f"./Policys/Perturbed_model_{i}.pth")       
    #     final_epoch,lower_bound = cb_sliding_window_search(traj_list,return_list,eval_policy,base_policy,min_epoch=min_epochs,max_epoch=max_epochs,step=step,converge=converge,confidence_level=confidence_level)
    #     lb_list.append(lower_bound)
    # end_time = time.time()  # Record the end time
    # elapsed_time = end_time - start_time  # Calculate elapsed time
    # converge_epoch_ls.append(final_epoch)
    eval_policy = torch.load(f"./Policys/Perturbed_model_{3}.pth")       
    final_epoch,lower_bound = cb_sliding_window_search(traj_list,return_list,eval_policy,base_policy,min_epoch=min_epochs,max_epoch=max_epochs,step=step,converge=converge,confidence_level=confidence_level)
    # lb_list.append(lower_bound)
    # print(f"Elapsed time: {elapsed_time:.4f} seconds")
            
    # np.savetxt("converge_record.txt", lb_list)
    plt.plot(final_epoch, lower_bound, marker='o', label='Thomas Lower Bound')
    plt.xlabel("Sample Size")
    plt.ylabel("Estimated Lower Bound (Normalized)")
    plt.title("90% Confidence Lower Bound")
    plt.grid(True)
    plt.legend()
    plt.show()