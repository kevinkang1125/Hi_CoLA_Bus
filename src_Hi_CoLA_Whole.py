import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import gym
import time
import copy
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
from src_clb_cal import *


if __name__ == "__main__":
    # Input parameters
    traj_num = 1000000
    gamma = 0.99
    pertub_size = 1000
    count = 0
    sigma = 0.4
    kl = 0.25 
    behavior_pi = Behavioural()
    behavior_pi.load_state_dict(torch.load("./Behavioural_model.pth"))
    env = gym.make('FlexibleBus-v0')
    start_time = time.time()  # Record the start time
    # Generate Trajectories using the behavior policy
    result_rec = []
    epoch_record = []
    done = False
    while not done:
        start_time = time.time()  # Record the start time
        traj_list,return_list = traj_collect(env = env,traj_num=traj_num,gamma=gamma,b_policy=behavior_pi)
        # Perturbation + KL_Divergence Threshold determination
        # Load data for KL calculation
        states = torch.tensor(traj_list.iloc[:60000, 0].values, dtype=torch.float32)
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
        for pi in perturbs:
            lb = Thomas_hc(behavior_pi,pi,traj_list,return_list,confidence_level=0.9)
            thomas_lbs.append(lb)

        
        # CoLA-Net Training
        input_dim = sum(p.numel() for p in behavior_pi.parameters())
        hi_cola = hc.train(features = perturbs, labels= thomas_lbs, model= Hi_CoLA_Net(input_dim=input_dim), lr=1e-6, weight_decay=1e-4, epochs=1000, verbose=True,logit=False,train_ratio=0.7)
        

        optimizer = torch.optim.Adam(behavior_pi.parameters(), lr=0.04)

        # Target KL divergence we want to achieve
        target_lb = torch.tensor([[1]], dtype=torch.float32)

        # Optimization loop
        loss_fn = nn.MSELoss()
        loss_trace, return_trace = [], []
        start_pi = copy.deepcopy(behavior_pi)
        output_pi = None
        for step in range(1200):
            optimizer.zero_grad()
            
            # Extract flattened weights from behavioural_model
            input_vector = get_flat_params(behavior_pi).unsqueeze(0)
            
            # Feed into the frozen KL regressor
            confidence_lb = hi_cola(input_vector)
            
            # Minimize difference to desired KL
            loss = loss_fn(confidence_lb, target_lb)
            loss.backward()
            optimizer.step()
            
            # Logging
            return_trace.append(confidence_lb.item())
            if kl_between_policies(behavior_pi,start_pi,states)>0.25:
                break
            output_pi = copy.deepcopy(behavior_pi)
        epoch_record.append(confidence_lb.item())
        behavior_pi = copy.deepcopy(output_pi)
        result_rec.extend(return_trace)
        if np.var(result_rec[-200:])<0.001*np.mean(result_rec[-200:]):
            print("Converged")
            done = True
    # Plotting        
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"train time per epoch: {elapsed_time:.4f} seconds")



    plt.plot(result_rec, label='Confidence Lowerbound')
    # plt.axhline(target_kl.item(), linestyle='--', color='r', label='Target KL')
    # plt.title("Confidence Lowerbound")
    plt.xlabel("Step",fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.grid(True)
    # # plt.tight_layout()
    plt.show()
    np.savetxt("Confidence_Lowerbound_Record.txt", result_rec, delimiter=",")
    np.savetxt("Epoch_Record.txt", epoch_record, delimiter=",")
    torch.save(behavior_pi.state_dict(), "./Optimized_Behavioural_model.pth")

