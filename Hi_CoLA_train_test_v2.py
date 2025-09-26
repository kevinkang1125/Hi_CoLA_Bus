import torch
import torch.multiprocessing as mp
import numpy as np
from multiprocessing import Pool, Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import gym
import copy
import flexible_bus
import json
import os
from functools import partial
import queue
from Behaviour_pi import Behavioural, Policy
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pandas as pd
import torch.nn as nn
from Simulation import traj_collect
from src_utils import *
from src_clb_cal_v2 import *

import os
os.environ['TORCH_MULTIPROCESSING_SHARING_STRATEGY'] = 'file_system'
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import torch.nn as nn




def load_model_features_and_labels(features, labels, train_ratio=0.7):
    """
    IMPROVED: Added input normalization for better training
    
    Args:
        features (list): list of pertubated models's weights
        labels (): confidence lower bound of the perturbated models
        normalize_inputs: Whether to normalize policy parameters (recommended: True)

    Returns:
        train_loader, test_loader: train and test dataloaders
        scaler: input scaler (for consistent preprocessing)
    """
    policy_params = [] 
    for model_path in features:
        model = model_path

        # Flatten all parameters to 1D feature vector
        flat_params = []
        for param in model.parameters():
            flat_params.append(np.array(param.detach().cpu().tolist()).flatten())
        feature_vector = np.concatenate(flat_params)

        policy_params.append(feature_vector)
    
    features = np.array(policy_params)
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Data shape: {X.shape[0]} policies, {X.shape[1]} parameters each")
    print(f"Thomas LB range: [{y.min():.4f}, {y.max():.4f}]")
    
    # IMPROVEMENT 1: Input normalization for policy parameters
    # scaler = None

    # print("Applying input normalization...")
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X = scaler.fit_transform(X)
    
    # IMPROVEMENT 2: Outlier removal for better training
    if len(y) > 20:  # Only if we have enough data
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
        X, y = X[mask], y[mask]
        print(f"After outlier removal: {len(X)} policies")
    
    X_tensor = torch.tensor(X, dtype=torch.float32)

    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    # Dataset and DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # IMPROVEMENT 3: Better batch size based on data size
    batch_size = min(32, max(8, len(train_set) // 10))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    return train_loader, val_loader, scaler


# === IMPROVED MODEL ===
class ImprovedHiCoLANet(nn.Module):
    """
    IMPROVED Hi-CoLA Network specifically designed for your 1,314-parameter policy
    
    Key improvements:
    1. Much deeper architecture to handle complexity
    2. Input normalization layer
    3. Better activations (GELU vs ReLU)
    4. Lower dropout rates
    5. Gradual dimension reduction instead of aggressive compression
    6. Proper weight initialization
    """
    def __init__(self, input_dim):
        super().__init__()
        
        print(f"Creating ImprovedHiCoLANet with {input_dim} input dimensions")
        
        # Input normalization layer (critical for high-dimensional inputs)
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Much deeper network with proper capacity
        self.net = nn.Sequential(
            # EXPAND first to capture parameter relationships
            nn.Linear(input_dim, 2048),      # 1314 -> 2048 (expand instead of compress)
            nn.LayerNorm(2048),
            nn.GELU(),                       # GELU better than ReLU for this task
            nn.Dropout(0.1),                 # Much lower dropout than original 0.3
            
            # Maintain high capacity
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # Gradual compression (vs original 1314->128 aggressive jump)
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 1024),           # Maintain capacity longer
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05),                # Even lower dropout near output
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            
            nn.Linear(128, 64),
            nn.GELU(),
            
            nn.Linear(64, 1),
            nn.Sigmoid()                     # Keep sigmoid for [0,1] bounds
        )
        
        # Better weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)


# === ORIGINAL MODEL (for comparison) ===
# class Hi_CoLA_Net(nn.Module):
#     """Original Hi-CoLA network (kept for comparison)"""
#     def __init__(self, input_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.net(x)


def train(features, labels, model, lr=5e-5, weight_decay=1e-5, epochs=2500, 
          verbose=True, logit=False, train_ratio=0.7, use_improvements=True):
    """
    IMPROVED training function with better hyperparameters and techniques
    
    Key improvements:
    1. Much higher learning rate (5e-5 vs 1e-6)
    2. More epochs (2500 vs 1000) 
    3. Better optimizer settings
    4. Learning rate scheduling
    5. Input normalization
    6. Gradient clipping
    7. Better early stopping
    """
    # Load Features and Labels with improvements
    train_loader, val_loader, scaler = load_model_features_and_labels(
        features, labels, train_ratio
    )
    
    # IMPROVEMENT 4: Better optimizer settings
    if use_improvements:
        print(f"Using improved training with lr={lr}, epochs={epochs}")
        optimizer = torch.optim.AdamW(  # AdamW instead of Adam
            model.parameters(), 
            lr=lr,                      # 50x higher than original 1e-6
            weight_decay=weight_decay,  # Lower weight decay
            betas=(0.9, 0.999)
        )
        
        # IMPROVEMENT 5: Learning rate scheduling
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=lr * 0.01
        )
    else:
        # Original settings for comparison
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
        scheduler = None
    
    # Better loss function for robustness
    if use_improvements and not logit:
        loss_fn = nn.SmoothL1Loss()  # More robust to outliers than MSE
    else:
        loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            
            # IMPROVEMENT 6: Gradient clipping for stability
            if use_improvements:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # IMPROVEMENT 7: Better early stopping with model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        # Early stopping with more patience
        max_patience = 150 if use_improvements else 50
        if patience > max_patience:
            print(f"Early stopping at epoch {epoch+1} (patience exceeded)")
            break
        
        if verbose:
            if epoch % 100 == 0 or epoch == epochs-1:
                lr_str = f", LR: {scheduler.get_last_lr()[0]:.2e}" if scheduler else ""
                print(f"Epoch {epoch+1:04d} | Train: {avg_train_loss:.6f} | "
                      f"Val: {avg_val_loss:.6f}{lr_str}")
        
        # Original early stopping condition (less effective)
        if not use_improvements and epoch > 200:
            if np.var(train_losses[-20:]) < 1e-2*np.mean(train_losses[-20:]):
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    # Visualization
    if verbose:
        plt.figure(figsize=(12, 4))
        
        # Training curves
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Learning Curve', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.yscale('log')  # Log scale for better visualization
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Final performance on validation set
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                y_true.extend(yb.squeeze().tolist())
                y_pred.extend(pred.squeeze().tolist())
        
        
        r2 = r2_score(y_true, y_pred)
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal (y = x)")
        plt.xlabel("Actual", fontsize=14)
        plt.ylabel("Predicted", fontsize=14)
        plt.title(f"Validation R² = {r2:.4f}", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    return model, train_losses, val_losses, scaler


def evaluate(features, labels, model, scaler=None, logit=False):
    """
    IMPROVED evaluation with proper preprocessing consistency
    """
    train_loader, val_loader, _ = load_model_features_and_labels(
        features, labels, train_ratio=0.8
    )
    
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            y_true.extend(yb.squeeze().tolist())
            y_pred.extend(pred.squeeze().tolist())
    
    # if logit:
    #     y_true = inv_logit(np.array(y_true))
    #     y_pred = inv_logit(np.array(y_pred))
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

    print(f"\nEvaluation Metrics on Validation Set:")
    print(f"MSE  = {mse:.6f}")
    print(f"MAE  = {mae:.6f}")  
    print(f"R²   = {r2:.4f}")
    print(f"MAPE = {mape:.2f}%")
    
    # Enhanced scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=30)
    
    # Better axis limits
    y_min, y_max = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    margin = (y_max - y_min) * 0.1
    plt.xlim(y_min - margin, y_max + margin)
    plt.ylim(y_min - margin, y_max + margin)
    
    plt.plot([y_min - margin, y_max + margin], [y_min - margin, y_max + margin], 
             'r--', label="Ideal (y = x)", linewidth=2)
    plt.xlabel("Actual", fontsize=14)
    plt.ylabel("Predicted", fontsize=14)
    plt.title(f"Predicted vs Actual\nR² = {r2:.4f}, MAPE = {mape:.2f}%", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return {'mse': mse, 'mae': mae, 'r2': r2, 'mape': mape}


def compare_models(features, labels, verbose=True):
    """
    Compare original vs improved Hi-CoLA architectures
    """
    print("=" * 60)
    print("COMPARING ORIGINAL vs IMPROVED Hi-CoLA")
    print("=" * 60)
    
    # Determine input dimension
    sample_params = []
    for param in features[0].parameters():
        sample_params.append(np.array(param.detach().cpu().tolist()).flatten())
    input_dim = len(np.concatenate(sample_params))
    
    print(f"Input dimension: {input_dim} parameters")
    
    results = {}
    
    print(f"\n2. Testing Improved Hi-CoLA...")
    improved_model = ImprovedHiCoLANet(input_dim)
    try:
        improved_model, _, _, scaler = train(
            features, labels, improved_model,
            lr=5e-5, epochs=2500, verbose=False
        )
        improved_metrics = evaluate(features, labels, improved_model, scaler)
        results['improved'] = improved_metrics
        print(f"   Improved R²: {improved_metrics['r2']:.4f}")
    except Exception as e:
        print(f"   Improved training failed: {e}")
        results['improved'] = {'r2': 0.0, 'mse': float('inf')}
    
    # # Summary
    # print(f"\n" + "=" * 60)
    # print("COMPARISON SUMMARY")
    # print("=" * 60)
    # for method, metrics in results.items():
    #     print(f"{method.capitalize():>12}: R² = {metrics['r2']:.4f}, MSE = {metrics.get('mse', 0):.6f}")
    
    # if results['improved']['r2'] > results['original']['r2']:
    #     improvement = (results['improved']['r2'] - results['original']['r2']) / max(abs(results['original']['r2']), 1e-6) * 100
    #     print(f"\nIMPROVEMENT: {improvement:.1f}% better R² with improved architecture")
        
    #     if results['improved']['r2'] > 0.6:
    #         print("✅ EXCELLENT: Improved Hi-CoLA successfully learns the mapping!")
    #     elif results['improved']['r2'] > 0.3:
    #         print("⚠️  MODERATE: Good improvement but room for further optimization")
    # else:
    #     print("❌ Improved version did not outperform original - check data quality")
    
    return results



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
        
        # def collect_chunk(args):
        #     env_name, pi_state_dict, chunk_size, gamma, seed = args
        #     # Create new environment and policy for each process
        #     env = gym.make(env_name)
        #     pi = Behavioural()
        #     pi.load_state_dict(pi_state_dict)
            
        #     # Set different seed for each process
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
            
        #     return traj_collect(env=env, traj_num=chunk_size, gamma=gamma, base_policy=pi)
        
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
                    futures.remove(future)
                
                # Limit concurrent futures
                if len(futures) > self.num_workers * 2:
                    # Wait for at least one to complete
                    next(as_completed(futures))
        
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
    
    def optimized_training_loop(self, behavior_pi, hi_cola, target_lb, env_name, traj_num, gamma, 
                               states, b_loop, max_steps=300, lr=0.004):
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
        output_pi = None
        
        for step in range(max_steps):
            optimizer.zero_grad()
            
            # Forward pass
            input_vector = get_flat_params(behavior_pi).unsqueeze(0)
            confidence_lb = hi_cola(input_vector)
            loss = loss_fn(confidence_lb, target_lb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            return_trace.append(confidence_lb.item())
            
            # Periodic evaluation (every 10 steps to reduce overhead)
            if step % 5 == 0:
                # Move to CPU for trajectory collection
                cpu_pi = copy.deepcopy(behavior_pi).cpu()
                
                # Use smaller trajectory collection during training
                traj_list, return_list = self.parallel_trajectory_collection(
                    env_name, cpu_pi, int(0.1 * traj_num), gamma, num_processes=2
                )
                np.savetxt(f"./exp_multi_epochs/dynamic_bounds/raw_reward_{b_loop}_{step}.txt", return_list, delimiter=",")
                # KL divergence check
            if kl_between_policies(behavior_pi, start_pi, test_states) > 0.25:
                break
            
            output_pi = copy.deepcopy(behavior_pi)
            print(f"Step {step+1}/{max_steps}, Loss: {loss.item():.4f}, Confidence LB: {confidence_lb.item():.4f}")
        return output_pi.cpu(), return_trace



def main():
    # Configuration
    traj_num = 200000#3
    gamma = 0.99
    pertub_size = 400
    target_count = 150
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
    stop_signal = 1
    
    # while not done:
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
    perturbs, final_bin_counts = trainer.parallel_perturbation_generation(
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
    print("Comparing Hi-CoLA model...")
    compare_models(perturbs, thomas_lbs)

if __name__ == "__main__":
    # Set multiprocessing start method
    if __name__ == "__main__":
        mp.set_start_method('spawn', force=True)
        
    # Set number of threads for PyTorch
    torch.set_num_threads(min(8, os.cpu_count()))
    
    main()