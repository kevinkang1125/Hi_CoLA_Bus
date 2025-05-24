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
    while not done:
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
        for pi in perturbs:
            lb = Thomas_hc(behavior_pi,pi,traj_list,return_list,confidence_level=0.9)
            thomas_lbs.append(lb)

        
        # CoLA-Net Training
        input_dim = sum(p.numel() for p in behavior_pi.parameters())
        hi_cola = hc.train(features = perturbs, labels= thomas_lbs, model= Hi_CoLA_Net(input_dim=input_dim), lr=1e-6, weight_decay=1e-4, epochs=1000, verbose=True,logit=False,train_ratio=0.7)
        
        # Policy Improvement
        # Flatten the Behavioural model's parameters into a vector (input to KLRegressor)
        def get_flat_params(model):
            return torch.cat([p.view(-1) for p in model.parameters()])

        # Setup optimizer to update the behavioural model's parameters
        optimizer = torch.optim.Adam(behavior_pi.parameters(), lr=0.04)

        # Target KL divergence we want to achieve
        target_lb = torch.tensor([[1]], dtype=torch.float32)

        # Optimization loop
        loss_fn = nn.MSELoss()
        loss_trace, kl_trace = [], []

        for step in range(1200):
            optimizer.zero_grad()
            
            # Extract flattened weights from behavioural_model
            input_vector = get_flat_params(behavior_pi).unsqueeze(0)
            
            # Feed into the frozen KL regressor
            kl_output = hi_cola(input_vector)
            
            # Minimize difference to desired KL
            loss = loss_fn(kl_output, target_lb)
            loss.backward()
            optimizer.step()
            
            # Logging
            loss_trace.append(loss.item())
            kl_trace.append(kl_output.item())
            
            if step % 50 == 0 or step == 299:
                print(f"Step {step:03d} | Loss: {loss.item():.6f} | KL: {kl_output.item():.4f}")
            base_pi = behavior_pi
            if kl_between_policies(base_pi,behavior_pi,states)>0.25:
                break
            pi_t = behavior_pi
        behavior_pi = pi_t
        done = True
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.plot(loss_trace)
    # plt.title("Loss")
    plt.xlabel("Step",fontsize = 20)
    plt.ylabel("Loss",fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.grid(True)



    plt.subplot(1, 2, 2)
    plt.plot(kl_trace)
    # plt.axhline(target_kl.item(), linestyle='--', color='r', label='Target KL')
    # plt.title("Confidence Lowerbound")
    plt.xlabel("Step",fontsize = 20)
    plt.ylabel("Predicted Confidence Lowerbound",fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.grid(True)

    # # plt.tight_layout()
    plt.show()
    torch.save(behavior_pi.state_dict(), "./Optimized_Behavioural_model.pth")

    print(f"Elapsed time: {elapsed_time:.4f} seconds")