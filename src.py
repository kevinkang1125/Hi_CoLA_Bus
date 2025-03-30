import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

import gym
import time
import flexible_bus
from policy import model_1, model_2  
import json
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def traj_simulate(seed,gamma,base_policy):
    
    """
    Func to Simulate the FRD process
        return: trajectory, return list
    
    """
    np.random.seed(seed)
    trajectory = {
        "observations": [],
        "actions": [],
        "return": 0.0,
        "policy": None
    }

    env = gym.make('FlexibleBus-v0')
    obs = env.reset()
    r = 0
    done = False
    trajectory["policy"] = base_policy

    while not done:
        trajectory["observations"].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)
        deviate_1 = int(np.random.choice([0, 1], p=[1-base_policy[0], base_policy[0]]))
        deviate_2 = int(np.random.choice([0, 1], p=[1-base_policy[1], base_policy[1]]))
        action = [deviate_1,deviate_2]  
        trajectory["actions"].append(action.tolist() if isinstance(action, np.ndarray) else action)
        obs, rewards, done, info = env.step(action)
        r = r * gamma + rewards
    trajectory["return"] = r
    
    return trajectory, r

def traj_save_json(trajectories, filepath):
    """Save trajectories to a JSON file
       Save reward list to txt file
    """
    def custom_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, int):  # Ensure integers remain integers
            return obj
        return str(obj)
    
    with open(filepath, 'w') as f:
        json.dump(trajectories, f, indent=4, default=custom_serializer)

def traj_collect(traj_num,gamma,base_policy,batch_size=1000):
    all_traj = []
    all_return = []
    traj_simulate_ = partial(traj_simulate, gamma=gamma, base_policy=base_policy)
    
    # Create directory dynamically based on `base_policy`
    folder_path = f"./Trajectories/{base_policy[0]}_{base_policy[1]}/"
    os.makedirs(folder_path, exist_ok=True)
    
    with ProcessPoolExecutor() as executor:
        for batch_start in tqdm(range(0, traj_num, batch_size), desc="Processing batches"):
            seeds = range(batch_start, min(batch_start + batch_size, traj_num))
            results = executor.map(traj_simulate_, seeds)
            for traj, ret in results:
                all_traj.append(traj)
                all_return.append(ret)

    # Save trajectories to JSON and return list to txt
    np.savetxt(f"{folder_path}/return_list.txt", all_return)
    traj_save_json(all_traj, f"{folder_path}/trajectories.json")
    return all_traj,all_return

def importance_sample(traj,eval_policy,base_policy,up_b,low_b):
    """
    Func to use the collected trajectories to estimate the evaluation policy

    Args:
        traj (_type_): dict
        eval_policy (_type_): _description_
        base_policy (_type_): _description_
        up_b (_type_): _description_
        low_b (_type_): _description_
    """
    eval_policy = [[1-eval_policy[0],eval_policy[0]],[1-eval_policy[1],eval_policy[1]]]
    base_policy = [[1-base_policy[0],base_policy[0]],[1-base_policy[1],base_policy[1]]]
    actions = traj["actions"]  # Assume it's a list of lists
    importance_weights = [
        (eval_policy[0][a[0]] * eval_policy[1][a[1]]) /
        (base_policy[0][a[0]] * base_policy[1][a[1]])
        for a in actions
    ]
    importance_weight = np.prod(importance_weights)
    norm_return = (traj["return"] - low_b) / (up_b - low_b)
    return importance_weight * norm_return

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
    max_return, min_return = max(return_list), min(return_list)
    folder_path = f"./Trajectories/{base_policy[0]}_{base_policy[1]}/"
    lower_bound_list, iters = [], []
    for epochs in tqdm(range(min_epoch, max_epoch, step), desc="Processing Epochs"):
        indices = np.random.choice(len(traj_list), size=epochs, replace=False)
        random_sample = [traj_list[idx] for idx in indices]
        importance_samples = []

        with Pool() as pool:
            importance_samples = list(
                tqdm(
                    pool.starmap(
                        importance_sample,
                        [(traj, eval_policy, base_policy, max_return, min_return) for traj in random_sample]
                    ),
                    total=epochs,
                    desc=f"Calculating Importance Sampling for {epochs} Epochs",
                    leave=False
                )
            )

        c_star, lower_bound = high_confidence_cal(importance_samples,confidence_level)
        #lower_bound_list_ = lower_bound_list
        lower_bound_list.append(lower_bound)
        iters.append(epochs)
        if epochs >10000:
            if np.var(lower_bound_list[-20:])<converge*np.mean(lower_bound_list[-20:]):
                break
        
    np.savetxt(f"{folder_path}/Lower_bound_list.txt", lower_bound_list)
    np.savetxt(f"{folder_path}/Sample_size_list.txt", iters)
    return epochs




if __name__ == "__main__":
    # Parameters
    max_epochs = 1000000
    step = 5000
    min_epochs = 1000
    alpha = 0.05  # Adjust based on desired sensitivity
    traj_num = 1000000
    gamma = 0.99
    converge = 0.0001
    confidence_level = 0.9
    # base_policy = [0.5,0.5]
    eval_policy = [0.1,0.9]
    converge_epoch_ls = []
    for i in tqdm(range(9),desc="Outer_loop"):
        for j in tqdm(range(9),desc="Inter_loop"): 
            start_time = time.time()  # Record the start time
            base_policy = [(i+1)*0.1,(j+1)*0.1]
            traj_list,return_list = traj_collect(traj_num=traj_num,gamma=gamma,base_policy=base_policy)
            final_epoch = cb_sliding_window_search(traj_list,return_list,eval_policy,base_policy,min_epoch=min_epochs,max_epoch=max_epochs,step=step,converge=converge,confidence_level=confidence_level)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            converge_epoch_ls.append(final_epoch)
            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            
    np.savetxt("converge_record.txt", converge_epoch_ls)
    
