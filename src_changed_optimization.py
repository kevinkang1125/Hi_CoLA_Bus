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
"""_summary_

This script to add the pertubation boundary as the early stops for the policy gradient training.
"""
# Fix NumPy compatibility issue
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
    
    def parallel_perturbation_generation(self, behavior_pi, states, sigma, kl_threshold, target_count, bin_counts):
        """Generate perturbed policies in parallel"""
        perturbs = []
        z_maxes = []
        manager = Manager()
        shared_bin_counts = manager.dict(bin_counts)
        result_queue = manager.Queue()
        
        def generate_perturb_batch(batch_size=50):
            batch_perturbs = []
            for _ in range(batch_size):
                pi_perturbed = perturb_add_v2(pi=copy.deepcopy(behavior_pi), c=sigma)
                kl_score = kl_between_policies(pi_perturbed, behavior_pi, states).item()
                
                if kl_score <= kl_threshold:
                    kl_range = get_kl_bin(kl_score)
                    if shared_bin_counts[kl_range] < target_count:
                        batch_perturbs.append((pi_perturbed, kl_range))
            
            return batch_perturbs
        
        # Use ThreadPoolExecutor for CPU-bound perturbation generation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            while not all(count >= target_count for count in shared_bin_counts.values()):
                # Submit batch jobs
                future = executor.submit(generate_perturb_batch)
                futures.append(future)
                
                # Process completed futures
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    batch_results = future.result()
                    for pi_perturbed, kl_range in batch_results:
                        if shared_bin_counts[kl_range] < target_count:
                            shared_bin_counts[kl_range] += 1
                            perturbs.append(pi_perturbed)
                            # z_maxes.extend(check_perturbation_range(pi_perturbed,base_pi=behavior_pi,c=sigma))
                    futures.remove(future)
                
                # Limit concurrent futures
                if len(futures) > self.num_workers * 2:
                    # Wait for at least one to complete
                    next(as_completed(futures))
        # np.savetxt(f"./exp_multi_epochs/dynamic_bounds/z_scores_dist.txt", z_maxes, delimiter=",")
        return perturbs, dict(shared_bin_counts)
    
    def parallel_thomas_calculation(self, behavior_pi, perturbs, traj_list, return_list, epochs, confidence_level):
        """Calculate Thomas LBs in parallel"""
        if isinstance(traj_list, dict):
            print("Converting traj_list from dict to DataFrame...")
            traj_list = pd.DataFrame(traj_list)
        # def calc_thomas_lb(perturbs):
        #     return batch_thomas_hc_v2(behavior_pi, perturbs, traj_list, return_list, 
        #                       epochs=epochs, confidence_level=confidence_level)
        
        # with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        #     # Submit all tasks
        #     futures = {executor.submit(calc_thomas_lb, pi): i for i, pi in enumerate(perturbs)}
        #     thomas_lbs = [0] * len(perturbs)
            
        #     # Collect results with progress bar
        #     for future in tqdm(as_completed(futures), total=len(perturbs), desc="Calculating Thomas LBs"):
        #         idx = futures[future]
        #         thomas_lbs[idx] = future.result()
        thomas_lbs = batch_thomas_hc_v2(
                behavior_pi=behavior_pi, 
                eval_policies=perturbs,  # ALL policies at once
                traj_list=traj_list, 
                return_list=return_list,
                epochs=epochs, 
                confidence_level=confidence_level,
                num_workers=self.num_workers,
                device=self.device
            )
        
        return thomas_lbs
    

    def optimized_training_loop_fatigue(self, behavior_pi, hi_cola, target_lb, env_name, traj_num, gamma, 
                                states, b_loop,threshold = 1, max_steps=100, lr=0.0001):
            """Optimized training loop with GPU utilization"""
            optimizer = torch.optim.Adam(behavior_pi.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            
            # Move models to GPU if available
            behavior_pi = behavior_pi.to(self.device)
            hi_cola = hi_cola.to(self.device)
            target_lb = target_lb.to(self.device)
            test_states = states.to(self.device)
            
            loss_trace, return_trace = [], []
            start_pi = copy.deepcopy(behavior_pi)
            output_pi = copy.deepcopy(behavior_pi)
            # kl_ls = []
            
            # Track best policy and its performance
            # best_mean_return = float('-inf')
            # best_pi = copy.deepcopy(behavior_pi)
            z_max_rc = []
            model_trace = []
            
            for step in range(max_steps):
                optimizer.zero_grad()
                # Forward pass
                input_vector = get_flat_params(behavior_pi).unsqueeze(0)
                train_pi = copy.deepcopy(behavior_pi)
                z_score_step = check_perturbation_range(train_pi,base_pi=start_pi,c=1.5)
                np.savetxt(f"./exp_multi_epochs/changed_object2/policy_differences_{b_loop}_{step}.txt", z_score_step, delimiter=",")
                confidence_lb = hi_cola(input_vector)
                return_trace.append(confidence_lb.item())
                model_trace.append(copy.deepcopy(behavior_pi).cpu())
                if step % 2 == 0:
                    # Move to CPU for trajectory collection
                    cpu_pi = copy.deepcopy(behavior_pi).cpu()
                    # Use smaller trajectory collection during training
                    traj_list, return_list = self.parallel_trajectory_collection(
                        env_name, cpu_pi, int(0.1 * traj_num), gamma, num_processes=2
                    )
                    np.savetxt(f"./exp_multi_epochs/changed_object2/raw_reward_{b_loop}_{step}.txt", return_list, delimiter=",")
                
                
                loss = -confidence_lb.squeeze()
                # Backward pass
                loss.backward()
                optimizer.step()
                if kl_between_policies(behavior_pi, start_pi, test_states).item() > 0.1:
                    break
                
                output_pi = copy.deepcopy(behavior_pi)
                print(f"Step {step+1}/{max_steps}, Loss: {loss.item():.4f}, Confidence LB: {confidence_lb.item():.4f}")
            
            return output_pi.cpu(), return_trace, model_trace

def main():
    # Configuration
    traj_num = 200000#3
    gamma = 1
    pertub_size = 400
    target_count = 200
    threshhold = 0.2
    sigma = 0.15
    kl = 0.3
    num_workers = min(8, os.cpu_count())  # Adjust based on your system
    
    # Initialize trainer and models
    trainer = OptimizedTrainer(num_workers=num_workers)
    
    behavior_pi = Behavioural()
    behavior_pi.load_state_dict(torch.load("./Behavioural_model.pth", map_location='cpu'))
    
    env_name = 'FlexibleBus-v0'
    
    # Training loop variables
    result_rec = []
    epoch_record = []
    step_rec = []
    done = False
    b_loop = 0
    stop_signal = 5
    mr_rec = []
    label_up_ls = []
    
    while not done:
        start_time = time.time()
        print(f"Starting behavioral loop {b_loop + 1}")
        
        # Parallel trajectory collection
        print("Collecting trajectories...")
        traj_list, return_list = trainer.parallel_trajectory_collection(
            env_name, behavior_pi, traj_num, gamma, num_processes=num_workers//2
        )
        
        # Prepare states tensor
        states = torch.tensor(
            np.stack(traj_list["obs_0"].iloc[:60000].values), 
            dtype=torch.float32
        )
        
        # Initialize bin counts
        bin_counts = {f"{round(start,1):.1f}-{round(start+0.1,1):.1f}": 0 
                     for start in np.arange(0, kl, 0.1)}
        
        # Parallel perturbation generation
        print("Generating perturbed policies...")
        perturbs, final_bin_counts= trainer.parallel_perturbation_generation(
            behavior_pi, states, sigma, kl, target_count, bin_counts
        )
        print(f"Generated {len(perturbs)} perturbed policies")
        print("Final bin counts:", final_bin_counts)
        
        # Parallel Thomas LB calculation
        print("Calculating Thomas lower bounds...")
        thomas_lbs = trainer.parallel_thomas_calculation(
            behavior_pi, perturbs, traj_list, return_list, 
            epochs=int(0.8 * traj_num), confidence_level=0.9
        )
        
        # Train Hi-CoLA model
        # print(f"similarity threshhold:{threshhold}")
        print("Training Hi-CoLA model...")
        # input_dim = sum(p.numel() for p in behavior_pi.parameters())
        sample_params = []
        for param in behavior_pi.parameters():
            sample_params.append(np.array(param.detach().cpu().tolist()).flatten())
        input_dim = len(np.concatenate(sample_params))
        hi_cola, _, _,scaler, mr, label_up = hc.train(
            features=perturbs,
            labels=thomas_lbs,
            model=Hi_CoLA_Net(input_dim=input_dim),
            lr=5e-5,          
            epochs=2500,      
            verbose=False
        )
        mr_rec.append(mr)
        label_up_ls.append(label_up)
        np.savetxt(f"./exp_multi_epochs/changed_object2/label_up_record.txt",label_up_ls, delimiter=",")
        hi_cola = HiCoLAWithScaler(hi_cola, scaler)
        # Optimized policy training
        print("Training policy...")
        target_lb = torch.tensor([[1]], dtype=torch.float32)
        base_pi = copy.deepcopy(behavior_pi)
        output_pi, return_trace, model_trace = trainer.optimized_training_loop_fatigue(
            behavior_pi, hi_cola, target_lb, env_name, traj_num, gamma, states,b_loop,threshhold
        )
        Thomas_gt = trainer.parallel_thomas_calculation(
            base_pi, model_trace, traj_list, return_list, 
            epochs=int(0.8 * traj_num), confidence_level=0.9
        )
        # Update records
        step_rec.append(len(return_trace))
        epoch_record.append(return_trace[-1] if return_trace else 0)
        behavior_pi = output_pi
        result_rec.extend(return_trace)
        
        # Save intermediate results
        os.makedirs("./exp_multi_epochs/changed_object2/", exist_ok=True)
        np.savetxt(f"./exp_multi_epochs/changed_object2/Confidence_Lowerbound_Thomas_{b_loop}.txt", 
                  Thomas_gt, delimiter=",")
        np.savetxt(f"./exp_multi_epochs/changed_object2/Confidence_Lowerbound_Predict_{b_loop}.txt", 
                  return_trace, delimiter=",")
        np.savetxt(f"./exp_multi_epochs/changed_object2/Epoch_Record.txt", 
                  epoch_record, delimiter=",")
        np.savetxt(f"./exp_multi_epochs/changed_object2/Epoch_Length_Record.txt", 
                  step_rec, delimiter=",")
        np.savetxt(f"./exp_multi_epochs/changed_object2/r2_Record.txt", 
                  mr_rec, delimiter=",")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Behavioral loop {b_loop + 1} completed in {elapsed_time:.2f} seconds")
        print(f"Confidence lowerbound: {epoch_record[-1]:.4f}")
        
        b_loop += 1
        if b_loop >= stop_signal:
            done = True
    
    # Final save
    torch.save(behavior_pi.state_dict(), 
              "./exp_multi_epochs/changed_object2/Optimized_Behavioural_model.pth")
    
    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(result_rec, label='Confidence Lowerbound')
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Confidence Lowerbound", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("./exp_multi_epochs/changed_object2/training_progress.png", dpi=300)
    plt.show()
    
    print(f"Training completed! Final results saved.")
    print(f"Total steps: {len(result_rec)}")
    print(f"Final confidence lowerbound: {result_rec[-1]:.4f}")

if __name__ == "__main__":
    # Set multiprocessing start method
    if __name__ == "__main__":
        mp.set_start_method('spawn', force=True)
        
    # Set number of threads for PyTorch
    torch.set_num_threads(min(8, os.cpu_count()))
    
    main()