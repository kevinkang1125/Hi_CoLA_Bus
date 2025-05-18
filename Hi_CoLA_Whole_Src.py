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
from Behaviour_pi import Behavioural, Policy
from src_Hi_CoLA import Hi_CoLA_Net
import src_Hi_CoLA as hc

import pandas as pd
import torch
import torch.nn as nn

from Simulation import traj_collect
from src_utils import *


if __name__ == "__main__":
    # Input parameters
    traj_num = 1000000
    gamma = 0.99
    pertub_size = 1000
    count = 0
    sigma = 0.4
    kl = 0.25 
    behavior_pi = Behavioural()
    behavior_pi.load_state_dict(torch.load("./Behavioural_model_2.pth"))
    env = gym.make('FlexibleBus-v0')
    start_time = time.time()  # Record the start time
    # Generate Trajectories using the behavior policy
    traj_list,return_list = traj_collect(env = env,traj_num=traj_num,gamma=gamma,b_policy=behavior_pi)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    # Perturbation + KL_Divergence Threshold determination
    states = torch.tensor(traj_list.iloc[:60000, 0].values, dtype=torch.float32)
    target_count = 200
    perturbs = []
    thomas_lbs = []
    while not count >= pertub_size:
        pi_perturbed = perturb_add(pi=behavior_pi, sigma=sigma)
        kl_score = kl_between_policies(pi_perturbed, behavior_pi, states).item()
        if kl_score < kl:
            count += 1
            save_path = f"./Policys/Whole_Loop/"
            os.makedirs(save_path, exist_ok=True)
            filename = f"Perturbed_model_{count}.pth"
            torch.save(pi_perturbed.state_dict(), os.path.join(save_path, filename))
            perturbs.append(pi_perturbed)
    
    # High Confidence Lower Bound Calculation
    thomas_lbs = []
    # CoLA-Net Training

    hi_cola = hc.train(features = perturbs, labels= thomas_lbs, model= Hi_CoLA_Net, lr=1e-6, weight_decay=1e-4, epochs=1000, verbose=True,logit=False,train_ratio=0.7)
    
    # Policy Improvement 
    print(f"Elapsed time: {elapsed_time:.4f} seconds")