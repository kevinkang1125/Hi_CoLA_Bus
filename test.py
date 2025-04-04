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

def traj_simulate(seed,gamma,base_policy):
    
    """
    Func to Simulate the FRD process
        return: trajectory, return list
    
    """
    # np.random.seed(seed)
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
        obs = torch.tensor(obs,dtype=torch.float32)
        [p1, p2] = base_policy(obs)
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
    

gamma = 0.99
behavior_pi = Behavioural(input_dim=5, hidden_dim=64, output_dim=2)
# 2. Load the weights into it
behavior_pi.load_state_dict(torch.load("./Behavioural_model.pth"))
trajectory, r = traj_simulate(seed=1, gamma=gamma, base_policy=behavior_pi)
# Ensure the key names match the output of traj_simulate
print(trajectory)