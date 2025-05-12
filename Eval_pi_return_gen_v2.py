import torch
import gym
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from Behaviour_pi import Behavioural, Policy
from functools import partial
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import flexible_bus
import os
import pandas as pd
import torch
import torch.nn as nn
# === Worker function (runs on GPU) ===
def simulate_batch_gpu(batch_seeds, gamma, policy_path, device="cuda"):
    model = torch.load(policy_path, map_location=device)
    model.to(device).eval()

    returns = []
    env = gym.make('FlexibleBus-v0')

    for seed in batch_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs = env.reset()
        r = 0
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            p1, p2 = model(obs_tensor)
            p1, p2 = float(p1), float(p2)

            deviate_1 = np.random.choice([0, 1], p=[1 - p1, p1])
            deviate_2 = np.random.choice([0, 1], p=[1 - p2, p2])
            action = [deviate_1, deviate_2]

            obs, reward, done, _ = env.step(action)
            r = r * gamma + reward

        returns.append(r)

    env.close()
    return returns


# === Master batch manager ===
def return_collect_gpu(traj_num, gamma, policy_path, batch_size=1000,num_workers=None):
    all_returns = []

    seeds = np.arange(traj_num)
    seed_batches = [seeds[i:i + batch_size] for i in range(0, traj_num, batch_size)]

    simulate_partial = partial(simulate_batch_gpu, gamma=gamma, policy_path=policy_path)

    # Use 'spawn' context to safely support CUDA
    with get_context("spawn").Pool(processes=num_workers) as pool:
        for batch_returns in tqdm(pool.imap(simulate_partial, seed_batches), total=len(seed_batches), desc="Simulating Batches on GPU"):
            all_returns.extend(batch_returns)

    
    return all_returns


if __name__ == "__main__":
    traj_num = 800000
    gamma = 0.9
    start_time = time.time()
    for m in range(2,5):
        bin = m*0.1
        folder_name = f"./Policys/{bin:.1f}-{bin+0.1:.1f}/"
        for pi_id in range(100):
            file_name = f"Perturbed_model_{pi_id}.pth"
            policy_path = os.path.join(folder_name, file_name)
            return_list = return_collect_gpu(traj_num, gamma, policy_path, batch_size=1000)
            np.savetxt(f"./Results/KL_Critical_Value_Check_EXP/{bin:.1f}-{bin+0.1:.1f}/return_list_{pi_id}.txt", return_list)
    elapsed = time.time() - start_time

    print(f"✅ Elapsed time (GPU): {elapsed:.2f} seconds")
