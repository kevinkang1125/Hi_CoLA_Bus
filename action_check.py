import torch
import torch.multiprocessing as mp
import numpy as np
from multiprocessing import Pool, Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import gym
import time
import copy
import flexible_bus
import json
import os
from functools import partial
import queue
from Behaviour_pi import Behavioural, Policy
from src_Hi_CoLA_v2 import Hi_CoLA_Net, HiCoLAWithScaler
import src_Hi_CoLA_v2 as hc

import pandas as pd
import torch.nn as nn
from Simulation import traj_collect
from src_utils import *
from src_clb_cal_v2 import *

import os
os.environ['TORCH_MULTIPROCESSING_SHARING_STRATEGY'] = 'file_system'

def collect_trajectory_chunk_global(args):
    """Global function that can be pickled for multiprocessing"""
    env_name, pi_state_dict, chunk_size, gamma, seed = args
    
    try:
        # Create new environment and policy for each process
        env = gym.make(env_name)
        pi = Behavioural()
        pi.load_state_dict(pi_state_dict)
        
        # Set different seed for each process  
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        return traj_collect(env=env, traj_num=chunk_size, gamma=gamma, base_policy=pi)
        
    except Exception as e:
        print(f"Worker process failed: {e}")
        # Return empty results to prevent crashes
        empty_df = pd.DataFrame({'trajectory_id': [], 'return': []})
        for i in range(5):
            empty_df[f'obs_{i}'] = []
            empty_df[f'direct_action_{i}'] = []
        return empty_df, []
    
class OptimizedTrainer:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def parallel_trajectory_collection(self, env_name, behavior_pi, traj_num, gamma, num_processes=4):
        """Collect trajectories in parallel"""
        chunk_size = traj_num // num_processes
        
        # Prepare arguments for each process
        pi_state_dict = behavior_pi.state_dict()
        args_list = [
            (env_name, pi_state_dict, chunk_size, gamma, i*1000)
            for i in range(num_processes)
        ]
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(collect_trajectory_chunk_global, args_list))
        
        # Combine results
        combined_traj = {}
        combined_returns = []
        
        for traj_list, return_list in results:
            combined_returns.extend(return_list)
            for key in traj_list.keys():
                if key not in combined_traj:
                    combined_traj[key] = traj_list[key]
                else:
                    combined_traj[key] = pd.concat([combined_traj[key], traj_list[key]], ignore_index=True)
        
        return combined_traj, combined_returns

def main():
    # Configuration
    os.makedirs("./exp_multi_epochs/changed_object2/", exist_ok=True)
    traj_num = 10000#3
    gamma = 1
    num_workers = min(8, os.cpu_count())  # Adjust based on your system
    
    # Initialize trainer and models
    trainer = OptimizedTrainer(num_workers=num_workers)
    
    behavior_pi = Behavioural()
    behavior_pi.load_state_dict(torch.load("./Behavioural_model.pth", map_location='cpu'))
    
    env_name = 'FlexibleBus-v0'
    
    traj_list, return_list = trainer.parallel_trajectory_collection(
        env_name, behavior_pi, traj_num, gamma, num_processes=num_workers//2
    )
    traj_list = pd.DataFrame(traj_list)
    traj_list.to_parquet("./exp_multi_epochs/changed_object2/trajectories.pq")
    



if __name__ == "__main__":
    # Set multiprocessing start method
    if __name__ == "__main__":
        mp.set_start_method('spawn', force=True)
        
    # Set number of threads for PyTorch
    torch.set_num_threads(min(8, os.cpu_count()))
    
    main()