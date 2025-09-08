import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

import gym
import time
import flexible_bus
from policy import model_1, model_2,model_3
import json
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)




def traj_simulate(seed,episode = 1000):
    
    """
    Func to Prepare data for Behaviuoral Policy Training
        return: State + Action Dataframe
    
    """
    np.random.seed(seed)
    trajectory_rows = []
    env = gym.make('FlexibleBus-v0')
    obs = env.reset()
    done = False
    trajectory_rows = []
    for i in range(episode):
        obs_list = obs.tolist() if isinstance(obs, np.ndarray) else obs
        action = model_3(obs)
        action_list = action.tolist() if isinstance(action, np.ndarray) else action
        row = obs_list + action_list  # total 7 values
        trajectory_rows.append(row)
        obs, rewards, done, info = env.step(action)
    # columns = [f"obs_{i}" for i in range(5)] + [f"act_{i}" for i in range(2)]
    # traj = pd.DataFrame(trajectory_rows, columns=columns)
    return trajectory_rows


def traj_collect(traj_num,batch_size=1000):
    """
    Multiprocessing to collect trajectories

    Args:
        traj_num (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 1000.

    Returns:
        all_traj: trajectory list in pandas dataframe
    """
    all_traj = []
    traj_simulate_ = partial(traj_simulate, episode= 10)    
    with ProcessPoolExecutor() as executor:
        for batch_start in tqdm(range(0, traj_num, batch_size), desc="Processing batches"):
            seeds = range(batch_start, min(batch_start + batch_size, traj_num))
            results = executor.map(traj_simulate_, seeds)
            for traj in results:
                all_traj.extend(traj)
    columns = [f"obs_{i}" for i in range(5)] + [f"act_{i}" for i in range(2)]
    traj = pd.DataFrame(all_traj, columns=columns)
    return traj



if __name__ == "__main__":
    # Parameters
    step = 5000
    min_epochs = 1000
    traj_num = 1000000
    start_time = time.time()  # Record the start time
    traj= traj_collect(traj_num=traj_num)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    traj.to_csv("traj.csv", index=False)
    
