import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import gym
import time
import flexible_bus
from policy import model_3
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

import pandas as pd
import torch
import torch.nn as nn

def traj_simulate(seed,gamma,base_policy):
    
    """
    Func to Simulate the FRD process
        return: trajectory, return list
    
    """
    np.random.seed(seed)
    trajectory = {
        "observations": [],
        "direct_actions":[],
        "actions": [],
        "return": 0.0
    }

    env = gym.make('FlexibleBus-v0')
    obs = env.reset()
    r = 0
    done = False
    
    while not done:
        trajectory["observations"].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)
        [p1, p2] = model_3(obs)
        direct_action = [p1, p2]
        trajectory["actions"].append(direct_action.tolist() if isinstance(direct_action, np.ndarray) else direct_action)
        deviate_1 = int(np.random.choice([0, 1], p=[1-p1, p1]))
        deviate_2 = int(np.random.choice([0, 1], p=[1-p2, p2]))
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

class Behavioural(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Parameters
    max_epochs = 1000000
    step = 5000
    min_epochs = 1000
    alpha = 0.05  # Adjust based on desired sensitivity
    traj_num = 1000000
    gamma = 0.99
    confidence_level = 0.9
    behavior_pi = Behavioural(input_dim=5, hidden_dim=64, output_dim=2)
    # 2. Load the weights into it
    behavior_pi.load_state_dict(torch.load("./Behavioural_model.pth"))
    for i in tqdm(range(9),desc="Outer_loop"):
        for j in tqdm(range(9),desc="Inter_loop"): 
            start_time = time.time()  # Record the start time
            traj_list,return_list = traj_collect(traj_num=traj_num,gamma=gamma,base_policy=behavior_pi)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")