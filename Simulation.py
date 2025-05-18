import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import gym
import time
import flexible_bus
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

import pandas as pd
import torch
import torch.nn as nn

def traj_simulate(seed,env, gamma,b_policy):
    
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

    obs = env.reset()
    r = 0
    done = False
    
    while not done:
        trajectory["observations"].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)
        obs = torch.tensor(obs,dtype=torch.float32)
        [p1, p2] = b_policy(obs)
        p1 = p1.item()
        p2 = p2.item()
        direct_action = [p1, p2]
        trajectory["direct_actions"].append(direct_action.tolist() if isinstance(direct_action, np.ndarray) else direct_action)
        deviate_1 = int(np.random.choice([0, 1], p=[1-p1, p1]))
        deviate_2 = int(np.random.choice([0, 1], p=[1-p2, p2]))
        action = [deviate_1,deviate_2]  
        trajectory["actions"].append(action.tolist() if isinstance(action, np.ndarray) else action)
        obs, rewards, done, info = env.step(action)
        r = r * gamma + rewards
    trajectory["return"] = r
    return trajectory, r


def traj_collect(env,traj_num, gamma, base_policy, batch_size=1000):
    all_rows = []
    all_return = []

    traj_simulate_ = partial(traj_simulate, env = env, gamma=gamma, base_policy=base_policy)

    with ProcessPoolExecutor() as executor:
        for batch_start in tqdm(range(0, traj_num, batch_size), desc="Processing batches"):
            seeds = range(batch_start, min(batch_start + batch_size, traj_num))
            results = executor.map(traj_simulate_, seeds)
            for traj_id, (traj, ret) in enumerate(results):
                observations = traj["observations"]
                direct_actions = traj["direct_actions"]
                actions = traj["actions"]
                row = {
                    "trajectory_id": batch_start + traj_id,
                    "return": ret
                }
                for t in range(len(observations)):
                    obs = observations[t]
                    da = direct_actions[t] 
                    a = actions[t]


                    # Flatten observation

                    row[f"obs_{t}"] = obs

                    row[f"action_{t}"] = da

                    row[f"direct_action_{t}"] = a


                    
                all_rows.append(row)

                all_return.append(ret)

    df = pd.DataFrame(all_rows)
    df.to_parquet("./Traject/trajectories_optimized.pq")
    np.savetxt("./return_list_optimized.txt", all_return)

    return df, all_return
