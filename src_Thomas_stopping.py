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
    
    def training_loop(self, behavior_pi, hi_cola, target_lb, env_name, traj_num, gamma, 
                                states, b_loop, traj_list, return_list,threshold=1, max_steps=100, lr=0.00005, patience=5):
        """
        Optimized training loop with Thomas LB monitoring for early stopping
        
        Args:
            thomas_check_interval: How often to compute actual Thomas LB
            patience: Number of consecutive drops before stopping
        """
        optimizer = torch.optim.Adam(behavior_pi.parameters(), lr=lr)
        
        # Move models to GPU if available
        behavior_pi = behavior_pi.to(self.device)
        hi_cola = hi_cola.to(self.device)
        target_lb = target_lb.to(self.device)
        test_states = states.to(self.device)
        
        # Initialize tracking variables
        return_trace = []
        model_trace = []
        thomas_lb_trace = []
        
        start_pi = copy.deepcopy(behavior_pi)
        # Best policy tracking
        best_thomas_lb = float('-inf')
        best_pi = None
        best_step = 0
        consecutive_drops = 0
        
        for step in range(max_steps):
            optimizer.zero_grad()
            # Forward pass
            input_vector = get_flat_params(behavior_pi).unsqueeze(0)
            # Get Hi-CoLA prediction for tracking
            hi_cola_lb = hi_cola(input_vector)
            return_trace.append(hi_cola_lb.item())
            model_trace.append(copy.deepcopy(behavior_pi).cpu())
            
            # Check actual Thomas lowerbound periodically
            # Calculate actual Thomas lowerbound
            cpu_pi = copy.deepcopy(behavior_pi).cpu()
            thomas_lb = self.parallel_thomas_calculation(
                copy.deepcopy(start_pi).cpu(),
                [cpu_pi],
                traj_list, 
                return_list,
                epochs=int(0.8 * traj_num),
                confidence_level=0.9
            )[0]
            
            thomas_lb_trace.append((step, thomas_lb))
            print(f"  Thomas LB: {thomas_lb:.4f} (best: {best_thomas_lb:.4f})")
            
            # Check if this is the best policy so far
            if thomas_lb > best_thomas_lb:
                best_thomas_lb = thomas_lb
                best_pi = copy.deepcopy(behavior_pi)
                best_step = step
                consecutive_drops = 0  # Reset counter when we improve
                print(f"  New best policy! Thomas LB: {best_thomas_lb:.4f}")
            else:
                # Thomas LB dropped
                consecutive_drops += 1
                print(f"  Thomas LB dropped. Consecutive drops: {consecutive_drops}/{patience}")
                
                if consecutive_drops >= patience:
                    print(f"Early stopping: Thomas LB dropped {patience} times consecutively")
                    print(f"Returning best policy from step {best_step} with Thomas LB: {best_thomas_lb:.4f}")
                    break
            
            # Collect trajectories periodically for monitoring
            if step % 10 == 0 and step > 0:
                cpu_pi = copy.deepcopy(behavior_pi).cpu()
                _, return_list_step = self.parallel_trajectory_collection(
                    env_name, cpu_pi, int(0.1 * traj_num), gamma, num_processes=2
                )
                np.savetxt(f"./exp_multi_epochs/changed_object5/raw_reward_{b_loop}_{step}.txt", 
                        return_list_step, delimiter=",")
            
            # Compute loss and backprop
            loss = -hi_cola_lb.squeeze()
            loss.backward()
            optimizer.step()
            
            # Check KL divergence constraint
            kl_div = kl_between_policies(behavior_pi, start_pi, test_states).item()
            if kl_div > 0.1:
                print(f"KL divergence {kl_div:.4f} > 0.1, stopping")
                break
        # Return the best policy found
        if best_pi is not None:
            output_pi = best_pi
            print(f"Returning best policy from step {best_step} with Thomas LB: {best_thomas_lb:.4f}")
        else:
            output_pi = copy.deepcopy(behavior_pi)
            print("No Thomas checks performed, returning current policy")
        
        # Save Thomas LB history

        np.savetxt(f"./exp_multi_epochs/changed_object5/thomas_lb_trace_{b_loop}.txt", 
                thomas_lb_trace, delimiter=",")
        
        return output_pi.cpu(), return_trace, model_trace

def main():
    # Configuration
    os.makedirs("./exp_multi_epochs/changed_object5/", exist_ok=True)
    traj_num = 200000#3
    gamma = 1
    pertub_size = 400
    target_count = 400
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
    stop_signal = 100
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
        np.savetxt(f"./exp_multi_epochs/changed_object5/label_up_record.txt",label_up_ls, delimiter=",")
        # hi_cola = HiCoLAWithScaler(hi_cola, scaler)
        # Optimized policy training
        print("Training policy...")
        target_lb = torch.tensor([[1]], dtype=torch.float32)
        base_pi = copy.deepcopy(behavior_pi)
        output_pi, return_trace, model_trace = trainer.training_loop(behavior_pi, hi_cola, target_lb, env_name, traj_num, gamma, 
                                states, b_loop, traj_list, return_list,max_steps=300, lr=0.00005, patience=10)
        Thomas_gt = trainer.parallel_thomas_calculation(
            base_pi, model_trace, traj_list, return_list, 
            epochs=int(0.8 * traj_num), confidence_level=0.9
        )
        # param_drift,scaler_o,scaler_n = analyze_scaling_drift(
        #     original_perturbs=perturbs, 
        #     optimized_policies=model_trace,
        #     labels=thomas_lbs
        # )
        # param_drift.to_csv(f"./exp_multi_epochs/changed_object5/parameter_drift_{b_loop}.csv", index=False)
        # visualize_scaling_drift(param_drift)
        # Update records
        step_rec.append(len(return_trace))
        epoch_record.append(return_trace[-1] if return_trace else 0)
        behavior_pi = output_pi
        result_rec.extend(return_trace)
        
        # Save intermediate results
        os.makedirs("./exp_multi_epochs/changed_object5/", exist_ok=True)
        np.savetxt(f"./exp_multi_epochs/changed_object5/Confidence_Lowerbound_Thomas_{b_loop}.txt", 
                  Thomas_gt, delimiter=",")
        np.savetxt(f"./exp_multi_epochs/changed_object5/Confidence_Lowerbound_Predict_{b_loop}.txt", 
                  return_trace, delimiter=",")
        np.savetxt(f"./exp_multi_epochs/changed_object5/Epoch_Record.txt", 
                  epoch_record, delimiter=",")
        np.savetxt(f"./exp_multi_epochs/changed_object5/Epoch_Length_Record.txt", 
                  step_rec, delimiter=",")
        np.savetxt(f"./exp_multi_epochs/changed_object5/r2_Record.txt", 
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
              "./exp_multi_epochs/changed_object5/Optimized_Behavioural_model.pth")
    
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
    plt.savefig("./exp_multi_epochs/changed_object5/training_progress.png", dpi=300)
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