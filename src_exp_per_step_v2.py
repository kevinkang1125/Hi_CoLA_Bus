import torch
import torch.multiprocessing as mp
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
import torch.nn as nn
from Simulation import traj_collect
from src_utils import *
from src_clb_cal import *


def main():

    traj_num = 300000
    gamma = 0.99
    pertub_size = 400
    count = 0
    sigma = 0.15
    kl = 0.3
    behavior_pi = Behavioural()
    behavior_pi.load_state_dict(torch.load("./Behavioural_model.pth"))
    env = gym.make('FlexibleBus-v0')

    start_time = time.time()
    result_rec = []
    epoch_record = []
    done = False
    b_loop = 0
    stop_signal = 2
    while not done:
        start_time = time.time()
        traj_list, return_list = traj_collect(env=env, traj_num=traj_num, gamma=gamma, base_policy=behavior_pi)
        states = torch.tensor(np.stack(traj_list["obs_0"].iloc[:60000].values), dtype=torch.float32)

        perturbs = []
        thomas_lbs = []
        count = 0
        while not count >= pertub_size:
            pi_perturbed = perturb_add(pi=behavior_pi, sigma=sigma)
            kl_score = kl_between_policies(pi_perturbed, behavior_pi, states).item()
            print(f"KL score: {kl_score:.4f}")
            if kl_score < kl:
                count += 1
                perturbs.append(copy.deepcopy(pi_perturbed))
                save_path = f"./Policys/Whole_Loop/"
                os.makedirs(save_path, exist_ok=True)
                filename = f"Perturbed_model_{count}.pth"
                torch.save(pi_perturbed.state_dict(), os.path.join(save_path, filename))

        for pi in tqdm(perturbs, desc="Calculating Thomas LBs"):
            lb = Thomas_hc(behavior_pi, pi, traj_list, return_list, epochs=int(0.8 * traj_num), confidence_level=0.9)
            thomas_lbs.append(lb)

        input_dim = sum(p.numel() for p in behavior_pi.parameters())
        hi_cola, _, _ = hc.train(
            features=perturbs,
            labels=thomas_lbs,
            model=Hi_CoLA_Net(input_dim=input_dim),
            lr=1e-6,
            weight_decay=1e-4,
            epochs=1000,
            verbose=False,
            logit=False,
            train_ratio=0.7
        )

        optimizer = torch.optim.Adam(behavior_pi.parameters(), lr=0.0001)
        target_lb = torch.tensor([[1]], dtype=torch.float32)

        loss_fn = nn.MSELoss()
        loss_trace, return_trace = [], []
        start_pi = copy.deepcopy(behavior_pi)
        output_pi = None
        for step in range(1200):
            optimizer.zero_grad()
            device = next(hi_cola.parameters()).device
            input_vector = get_flat_params(behavior_pi).unsqueeze(0).to(device)
            confidence_lb = hi_cola(input_vector)
            loss = loss_fn(confidence_lb, target_lb)
            loss.backward()
            optimizer.step()
            return_trace.append(confidence_lb.item())

            device = next(behavior_pi.parameters()).device
            start_pi = start_pi.to(device)
            test_states = states.to(device)
            traj_list, return_list = traj_collect(env=env, traj_num=traj_num, gamma=gamma, base_policy=copy.deepcopy(behavior_pi).cpu())
            np.savetxt(f"./exp_per_step/lr_0.0001/raw_reward_{b_loop}_{step}.txt", return_list, delimiter=",")
            if kl_between_policies(behavior_pi, start_pi, test_states) > 0.25:
                break
            output_pi = copy.deepcopy(behavior_pi)
            # traj_list, return_list = traj_collect(env=env, traj_num=traj_num, gamma=gamma, base_policy=copy.deepcopy(output_pi))
            # np.savetxt(f"./exp_per_step/raw_reward_{b_loop}_{step}.txt", return_list, delimiter=",")

        epoch_record.append(confidence_lb.item())
        behavior_pi = copy.deepcopy(output_pi)
        result_rec.extend(return_trace)
        if np.var(result_rec[-200:]) < 0.001 * np.mean(result_rec[-200:]):
            print("Converged")
            done = True

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"train time per epoch: {elapsed_time:.4f} seconds")
        b_loop += 1
        if b_loop >= stop_signal:
            done = True
        print(f"Behavioral Loop {b_loop} finished, confidence lowerbound: {confidence_lb.item():.4f}")
        behavior_pi = copy.deepcopy(behavior_pi).cpu()
        np.savetxt("./exp_per_step/lr_0.0001/Confidence_Lowerbound_Record.txt", result_rec, delimiter=",")
        np.savetxt("./exp_per_step/lr_0.0001/Epoch_Record.txt", epoch_record, delimiter=",")

    # Plotting
    import matplotlib.pyplot as plt
    plt.plot(result_rec, label='Confidence Lowerbound')
    plt.xlabel("Step", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.show()

    np.savetxt("./exp_per_step/lr_0.0001/Confidence_Lowerbound_Record.txt", result_rec, delimiter=",")
    np.savetxt("./exp_per_step/lr_0.0001/Epoch_Record.txt", epoch_record, delimiter=",")
    torch.save(behavior_pi.state_dict(), "./exp_per_step/lr_0.0001/Optimized_Behavioural_model.pth")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
