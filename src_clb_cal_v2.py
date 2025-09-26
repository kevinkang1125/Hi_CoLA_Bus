import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from Behaviour_pi import Behavioural, Policy
import gym
import time
import json
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from threading import Lock

# Global GPU lock for thread safety
gpu_lock = Lock()

def batch_importance_sample_global(traj_df, eval_policies, behave_pi, up_b, low_b, traj_len=5, device="cuda"):
    """
    Batched importance sampling for multiple evaluation policies
    Much more efficient than calling importance_sample_global multiple times
    """
    with gpu_lock:  # Ensure thread safety for GPU operations
        # Move behavioral policy to GPU once
        behave_pi.to(device)
        behave_pi.eval()
        
        all_obs = []
        all_actions = []
        return_values = traj_df["return"].to_numpy()

        # Preload and flatten obs/actions (do this once)
        for _, row in traj_df.iterrows():
            for i in range(traj_len):
                all_obs.append(row[f"obs_{i}"])
                all_actions.append(row[f"direct_action_{i}"])

        # Convert to tensors on GPU once
        obs_tensor = torch.tensor(all_obs, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long, device="cpu")
        
        # Get behavioral policy output once
        with torch.no_grad():
            behave_out = behave_pi(obs_tensor)
        
        # Convert behavioral output to probabilities
        p1_base, p2_base = behave_out[:, 0], behave_out[:, 1]
        base_prob_1 = torch.stack([1 - p1_base, p1_base], dim=1)
        base_prob_2 = torch.stack([1 - p2_base, p2_base], dim=1)
        
        # Get action indices
        a1 = actions_tensor[:, 0]
        a2 = actions_tensor[:, 1]
        
        # Calculate base action probabilities once
        base_action_probs = (base_prob_1[range(len(a1)), a1] * 
                           base_prob_2[range(len(a2)), a2]).cpu()
        
        # Batch process all evaluation policies
        batch_results = []
        
        # Process policies in smaller batches to manage GPU memory
        batch_size = min(8, len(eval_policies))  # Adjust based on GPU memory
        
        for i in range(0, len(eval_policies), batch_size):
            batch_policies = eval_policies[i:i+batch_size]
            batch_iws = []
            
            for eval_pi in batch_policies:
                eval_pi.to(device)
                eval_pi.eval()
                
                with torch.no_grad():
                    eval_out = eval_pi(obs_tensor)
                
                # Convert eval output to probabilities
                p1_eval, p2_eval = eval_out[:, 0], eval_out[:, 1]
                eval_prob_1 = torch.stack([1 - p1_eval, p1_eval], dim=1)
                eval_prob_2 = torch.stack([1 - p2_eval, p2_eval], dim=1)
                
                # Calculate evaluation action probabilities
                eval_action_probs = (eval_prob_1[range(len(a1)), a1] * 
                                   eval_prob_2[range(len(a2)), a2]).cpu()
                
                # Importance weights
                iw = (eval_action_probs / base_action_probs).cpu().tolist()
                traj_iws = np.array(iw).reshape(-1, traj_len)  # Need to convert back to numpy
                traj_weights = traj_iws.prod(axis=1)
                                
                # Normalize returns
                norm_returns = (return_values - low_b) / (up_b - low_b)
                wr_list = norm_returns * traj_weights
                
                batch_iws.append(wr_list.tolist())
                
                # Move policy back to CPU to free GPU memory
                eval_pi.cpu()
            
            batch_results.extend(batch_iws)
            
            # Clear GPU cache after each batch
            if device == "cuda":
                torch.cuda.empty_cache()
        
        # Move behavioral policy back to CPU
        behave_pi.cpu()
        
    return batch_results

def optimized_high_confidence_cal(sample, confidence_level=0.9, tol=1e-3, seed=None):
    """
    Optimized high confidence calculation with proper random seeding
    """
    if seed is not None:
        np.random.seed(seed)
    
    delta = 1 - confidence_level
    sample = np.array(sample)
    sample_size = max(len(sample) // 20, 1)  # Ensure at least 1 sample
    
    if len(sample) <= sample_size:
        random_sample = sample.copy()
    else:
        random_sample = np.random.choice(sample, size=sample_size, replace=False)
    
    iw_returns = random_sample
    n = len(random_sample)
    
    if n <= 1:
        return 1.0, np.mean(sample)  # Fallback for small samples
    
    c_min, c_max = 1, 50
    best_c, best_lower_bound = c_min, -float('inf')

    while (c_max - c_min) > tol:
        c_mid = (c_min + c_max) / 2
        truncated_returns = np.minimum(iw_returns, c_mid)

        empirical_mean = np.mean(truncated_returns)

        term_1 = empirical_mean
        term_2 = 7 * c_mid * np.log(2 / delta) / (3 * (n - 1))
        
        # Handle potential numerical issues
        variance_term = (n * np.sum((truncated_returns / c_mid) ** 2) - 
                        (np.sum(truncated_returns / c_mid)) ** 2)
        if variance_term < 0:
            variance_term = 0
            
        term_3 = np.sqrt(
            (2 * np.log(2 / delta) / ((n - 1) * n * (len(sample) - n + 1))) *
            variance_term
        )
        lower_bound = term_1 - term_2 - term_3

        if lower_bound > best_lower_bound:
            best_lower_bound = lower_bound
            best_c = c_mid

        if lower_bound > best_lower_bound:
            c_min = c_mid
        else:
            c_max = c_mid
    
    # Final calculation with full sample
    m = len(sample)
    truncated_returns = np.minimum(sample, best_c)
    pairwise_sum = 2 * (m**2) * np.var(truncated_returns, ddof=0)

    term_1 = np.mean(truncated_returns)
    term_2 = 7 * best_c * np.log(2 / delta) / (3 * (m - 1))
    term_3 = np.sqrt((np.log(2 / delta)) * pairwise_sum / (m - 1)) / m
    lower_bound = term_1 - term_2 - term_3
    
    return best_c, lower_bound

def batch_thomas_hc_v2(behavior_pi, eval_policies, traj_list, return_list, epochs, 
                       confidence_level=0.9, num_workers=16, device="cuda"):
    """
    Optimized batch processing of Thomas HC for multiple evaluation policies
    """
    # Pre-process trajectory data once
    return_array = traj_list.iloc[:, 1].to_numpy()
    max_return, min_return = 400, 200
    #max_return, min_return = max(return_list), min(return_list)
    
    # Sample trajectory IDs once (all policies use same sample)
    sampled_ids = np.random.choice(len(traj_list), size=epochs, replace=False)
    sampled_df = traj_list[traj_list["trajectory_id"].isin(sampled_ids)]
    
    print(f"Computing importance samples for {len(eval_policies)} policies...")
    
    # Batch compute importance samples (GPU-intensive part)
    all_importance_samples = batch_importance_sample_global(
        sampled_df, eval_policies, behavior_pi, max_return, min_return, device=device
    )
    
    print("Computing confidence bounds in parallel...")
    
    # Parallel computation of confidence bounds (CPU-intensive part)
    def compute_single_bound(args):
        importance_samples, seed = args
        return optimized_high_confidence_cal(importance_samples, confidence_level, seed=seed)
    
    # Create arguments with different seeds for each worker
    args_list = [(samples, i * 1000) for i, samples in enumerate(all_importance_samples)]
    
    # Use ThreadPoolExecutor for CPU-bound confidence calculation
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(compute_single_bound, args_list),
            total=len(args_list),
            desc="Computing confidence bounds"
        ))
    
    # Extract just the lower bounds
    lower_bounds = [result[1] for result in results]
    
    return lower_bounds

# Backward compatibility wrapper
def Thomas_hc_v2_optimized(behavior_pi, eval_policy, traj_list, return_list, epochs, confidence_level=0.9):
    """
    Drop-in replacement for Thomas_hc_v2 with single policy
    """
    results = batch_thomas_hc_v2(behavior_pi, [eval_policy], traj_list, return_list, 
                                epochs, confidence_level, num_workers=1)
    return results[0]

# Original functions for comparison/fallback
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

def high_confidence_cal(sample, confidence_level=0.9, tol=1e-3):
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

def Thomas_hc_v2(behavior_pi, eval_policy, traj_list, return_list, epochs, confidence_level=0.9):
    # Parameters
    # epochs = 200000
    base_policy = behavior_pi
    return_list = traj_list.iloc[:, 1].to_numpy()
    max_return, min_return = 60, 0
    sampled_ids = np.random.choice(len(traj_list), size=epochs, replace=False)
    sampled_df = traj_list[traj_list["trajectory_id"].isin(sampled_ids)]
    importance_samples = importance_sample_global(
        sampled_df, eval_policy, base_policy, max_return, min_return
    )
    c_star, lower_bound = high_confidence_cal(importance_samples,confidence_level) 

    return lower_bound