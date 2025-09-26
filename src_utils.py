import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from Behaviour_pi import Behavioural, Policy
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import pandas as pd



def kl_between_policies(pi_perturbed, pi, states,eps=1e-8):
    with torch.no_grad():
        probs_pi = pi(states)
        probs_pert = pi_perturbed(states)
        p = torch.clamp(probs_pi, eps, 1 - eps)
        q = torch.clamp(probs_pert, eps, 1 - eps)
        # kl_vals = bernoulli_kl(probs_pert, probs_pi)
        #kl_vals = (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).sum(dim=-1)
        # kl_vals = (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q)))
        # kl = kl_vals.mean(dim=-1)
        kl_per_dim = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))  # [batch_size, 2]
        # Total KL is SUM across dimensions (not mean!)
        kl_total = kl_per_dim.sum(dim=-1)  # [batch_size] ✅ CORRECT
        # Then average across batch
        kl = kl_total.mean()  # Scalar
    return kl

def check_adaptive_perturbation_escape(current_pi, base_pi, c=0.15, threshold=3.0):
    """
    Direct check based on your perturbation formula
    
    Returns:
        escaped: bool - True if outside perturbation space
        max_z: float - Maximum deviation in units of adaptive sigma
        layer_info: dict - Per-layer statistics
    """
    layer_info = {}
    global_max_z = 0
    
    base_params = list(base_pi.parameters())
    current_params = list(current_pi.parameters())
    
    for idx, (base_p, curr_p) in enumerate(zip(base_params, current_params)):
        if base_p.requires_grad:
            # Your adaptive sigma formula
            sigma = torch.sqrt(c * base_p.abs() + 1e-8)
            
            # How many sigmas is current from base?
            z_scores = torch.abs(curr_p - base_p) / sigma
            
            max_z = z_scores.max().item()
            mean_z = z_scores.mean().item()
            
            layer_info[f'layer_{idx}'] = {
                'max_z': max_z,
                'mean_z': mean_z,
                'escaped': max_z > threshold
            }
            
            global_max_z = max(global_max_z, max_z)
    
    escaped = global_max_z > threshold
    
    return escaped,global_max_z

def check_perturbation_range(current_pi, base_pi, c=0.15):
    """
    Direct check based on your perturbation formula
    
    Returns:
        escaped: bool - True if outside perturbation space
        max_z: float - Maximum deviation in units of adaptive sigma
        layer_info: dict - Per-layer statistics
    """
    # global_max_z = 0
    base_flat = torch.cat([p.view(-1) for p in base_pi.parameters() if p.requires_grad])
    curr_flat = torch.cat([p.view(-1) for p in current_pi.parameters() if p.requires_grad])
    
    # Element-wise adaptive sigma (same as in perturb_add_v2)
    sigma = torch.sqrt(c * base_flat.abs() + 1e-8)
    
    # Element-wise z-scores
    z_scores = torch.abs(curr_flat - base_flat) / sigma
    z_list = z_scores.tolist()
    
    # base_params = list(base_pi.parameters())
    # current_params = list(current_pi.parameters())
    
    # for idx, (base_p, curr_p) in enumerate(zip(base_params, current_params)):
    #     if base_p.requires_grad:
    #         # Your adaptive sigma formula
    #         sigma = torch.sqrt(c * base_p.abs() + 1e-8)
            
    #         # How many sigmas is current from base?
    #         z_scores = torch.abs(curr_p - base_p) / sigma
            
    #         max_z = z_scores.max().item()
    #         mean_z = z_scores.mean().item()
            
    #         # layer_info[f'layer_{idx}'] = {
    #         #     'max_z': max_z,
    #         #     'mean_z': mean_z,
    #         #     'escaped': max_z > threshold
    #         # }
            
    #         global_max_z = max(global_max_z, max_z)
    
    # escaped = global_max_z > threshold
    
    return z_list

def check_adaptive_perturbation_escape_percent(current_pi, base_pi, c=0.15, threshold=3.0):
    """
    Check if any individual element has escaped the perturbation space.
    Uses flattened parameters to match Hi-CoLA training format.
    
    Returns:
        escaped: bool - True if any element is outside perturbation space
        max_z: float - Maximum deviation across all elements in units of adaptive sigma
        stats: dict - Statistics about the escape
    """
    # Flatten all parameters - matching your Hi-CoLA training format
    base_flat = torch.cat([p.view(-1) for p in base_pi.parameters() if p.requires_grad])
    curr_flat = torch.cat([p.view(-1) for p in current_pi.parameters() if p.requires_grad])
    
    # Element-wise adaptive sigma (same as in perturb_add_v2)
    sigma = torch.sqrt(c * base_flat.abs() + 1e-8)
    
    # Element-wise z-scores
    z_scores = torch.abs(curr_flat - base_flat) / sigma
    
    
    # Find max deviation across ALL elements
    max_z = z_scores.max().item()
    
    # Additional statistics
    stats = {
        'max_z': max_z,
        'mean_z': z_scores.mean().item(),
        'std_z': z_scores.std().item(),
        'num_escaped': (z_scores > threshold).sum().item(),
        'total_params': z_scores.numel(),
        'percent_escaped': 100.0 * (z_scores > threshold).sum().item() / z_scores.numel()
    }
    
    # Any single element escape means policy has escaped
    escaped = max_z > threshold
    
    return escaped, max_z, stats

def check_adaptive_perturbation_escape_75(current_pi, base_pi, c=0.15, threshold=0.03):
    """
    Simplified version that checks if 75th percentile of z-scores exceeds threshold.
    
    Returns:
        escaped: bool - True if 75th percentile exceeds threshold
        q75_z: float - The 75th percentile z-score
        stats: dict - Basic statistics
    """
    # Flatten all parameters
    base_flat = torch.cat([p.view(-1) for p in base_pi.parameters() if p.requires_grad])
    curr_flat = torch.cat([p.view(-1) for p in current_pi.parameters() if p.requires_grad])
    
    # Element-wise adaptive sigma
    sigma = torch.sqrt(c * base_flat.abs() + 1e-8)
    
    # Element-wise z-scores
    z_scores = torch.abs(curr_flat - base_flat) / sigma
    
    # Calculate 75th percentile
    q75_z = torch.quantile(z_scores, 0.75).item()
    
    # Basic statistics
    stats = {
        'q75_z': q75_z,
        'max_z': z_scores.max().item(),
        'mean_z': z_scores.mean().item(),
        'percent_above_threshold': 100.0 * (z_scores > threshold).sum().item() / z_scores.numel()
    }
    
    # 75th percentile escape check
    escaped = q75_z > threshold
    
    return escaped, q75_z

def perturb_add_v2(pi, c=0.3):
    """
    Perturb the policy parameters with Gaussian noise where sigma is proportional to each weight's magnitude.

    Args:
        pi (Policy): the original policy network
        c (float): proportional constant to scale sigma = c * |weight|

    Returns:
        pi_perturbed (Policy): the perturbed policy
    """
    pi_perturbed = Policy()
    pi_perturbed.load_state_dict(pi.state_dict())

    with torch.no_grad():
        for param in pi_perturbed.parameters():
            if param.requires_grad:
                sigma = torch.sqrt(c * param.abs() + 1e-8)
                noise = torch.randn_like(param) * sigma
                param.add_(noise)

    return pi_perturbed

def perturb_add(pi, sigma=0.3):
    pi_perturbed = Policy()
    pi_perturbed.load_state_dict(pi.state_dict())
    with torch.no_grad():
        for param in pi_perturbed.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * sigma
                param.add_(noise)
    return pi_perturbed

def get_flat_params(model):
    # Flatten the Behavioural model's parameters into a vector
    return torch.cat([p.view(-1) for p in model.parameters()])

def get_kl_bin(kl_score):
    bins = np.arange(0, 0.3, 0.1)
    for start in bins:
        end = round(start + 0.1, 1)
        if start <= kl_score < end:
            return f"{start:.1f}-{end:.1f}"
    return None


# def kl_filter(kl_score,kl_threshold=0.3):
#     # target_count = 200
#     # bin_counts = {f"{round(start,1):.1f}-{round(start+0.1,1):.1f}": 0 for start in np.arange(0, 0.3, 0.1)}


#     # while not all(count >= target_count for count in bin_counts.values()):

#     #     if kl_score < kl_threshold:
#     #         folder = get_kl_bin(kl_score)
#     #         if folder and bin_counts[folder] < target_count:
#     #             bin_counts[folder] += 1

def analyze_scaling_drift(original_perturbs, optimized_policies,labels):
    """Check how much the scaling parameters change when adding optimized policies"""
    policy_params = [] 
    for model in original_perturbs:
        # Flatten all parameters to 1D feature vector
        flat_params = []
        for param in model.parameters():
            flat_params.append(np.array(param.detach().cpu().tolist()).flatten())
        feature_vector = np.concatenate(flat_params)

        policy_params.append(feature_vector)
    
    features = np.array(policy_params)
    X = np.array(features)
    y = np.array(labels)
    
    # IMPROVEMENT 1: Input normalization for policy parameters
    # scaler = None

    # print("Applying input normalization...")
    # scaler = RobustScaler()  # More robust to outliers than StandardScaler
    # X = scaler.fit_transform(X)
    
    # IMPROVEMENT 2: Outlier removal for better training
    if len(y) > 20:  # Only if we have enough data
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
        X, y = X[mask], y[mask]
        # print(f"After outlier removal: {len(X)} policies")
    
    X_original = X
    # Get original training data
    # X_original = np.array([get_flat_params(pi).numpy() for pi in original_perturbs])
    
    # Fit scaler on original data only
    scaler_original = RobustScaler()
    scaler_original.fit(X_original)
    
    # Now add optimized policies one by one and check drift
    drift_results = []
    X_accumulated = X_original.copy()
    
    for i, opt_pi in enumerate(optimized_policies):
        # Add this optimized policy
        flat_params_opt = []
        for param in opt_pi.parameters():
            flat_params_opt.append(np.array(param.detach().cpu().tolist()).flatten())
        opt_params = np.concatenate(flat_params_opt).reshape(1, -1)
        # opt_params = get_flat_params(opt_pi).numpy().reshape(1, -1)
        X_accumulated = np.vstack([X_accumulated, opt_params])
        
        # Fit new scaler on accumulated data
        scaler_new = RobustScaler()
        scaler_new.fit(X_accumulated)
        
        # Calculate drift in scaling parameters
        median_drift = np.abs(scaler_new.center_ - scaler_original.center_)
        iqr_drift = np.abs(scaler_new.scale_ - scaler_original.scale_)
        
        # Relative drift (percentage)
        median_drift_pct = median_drift / (np.abs(scaler_original.center_) + 1e-8) * 100
        iqr_drift_pct = iqr_drift / (scaler_original.scale_ + 1e-8) * 100
        
        drift_results.append({
            'step': i,
            'median_drift_max': np.max(median_drift),
            'median_drift_mean': np.mean(median_drift),
            'iqr_drift_max': np.max(iqr_drift),
            'iqr_drift_mean': np.mean(iqr_drift),
            'median_drift_pct_max': np.max(median_drift_pct),
            'iqr_drift_pct_max': np.max(iqr_drift_pct),
            'params_outside_3iqr': check_outlier_params(opt_params, scaler_original)
        })
    
    return pd.DataFrame(drift_results), scaler_original, scaler_new

def check_outlier_params(params, scaler):
    """Check which parameters are outliers relative to original distribution"""
    # Transform using original scaler
    params_scaled = scaler.transform(params.reshape(1, -1))[0]
    
    # Count how many parameters are > 3 IQR from median (outliers)
    outliers = np.abs(params_scaled) > 3
    
    return {
        'n_outliers': np.sum(outliers),
        'pct_outliers': np.mean(outliers) * 100,
        'max_deviation': np.max(np.abs(params_scaled)),
        'outlier_indices': np.where(outliers)[0][:10]  # First 10 outlier indices
    }
    

def visualize_scaling_drift(drift_df, param_names=None):
    """Visualize how scaling parameters drift during optimization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot 1: Median drift over optimization steps
    axes[0, 0].plot(drift_df['step'], drift_df['median_drift_mean'], label='Mean')
    axes[0, 0].plot(drift_df['step'], drift_df['median_drift_max'], label='Max')
    axes[0, 0].set_xlabel('Optimization Step')
    axes[0, 0].set_ylabel('Median Drift (absolute)')
    axes[0, 0].set_title('Drift in Median (Center)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: IQR drift over optimization steps
    axes[0, 1].plot(drift_df['step'], drift_df['iqr_drift_mean'], label='Mean')
    axes[0, 1].plot(drift_df['step'], drift_df['iqr_drift_max'], label='Max')
    axes[0, 1].set_xlabel('Optimization Step')
    axes[0, 1].set_ylabel('IQR Drift (absolute)')
    axes[0, 1].set_title('Drift in IQR (Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Percentage of outlier parameters
    axes[0, 2].plot(drift_df['step'], 
                    drift_df['params_outside_3iqr'].apply(lambda x: x['pct_outliers']))
    axes[0, 2].set_xlabel('Optimization Step')
    axes[0, 2].set_ylabel('% Parameters Outside 3 IQR')
    axes[0, 2].set_title('Parameters Becoming Outliers')
    axes[0, 2].axhline(y=5, color='r', linestyle='--', label='5% threshold')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot 4: Relative drift (percentage)
    axes[1, 0].plot(drift_df['step'], drift_df['median_drift_pct_max'])
    axes[1, 0].set_xlabel('Optimization Step')
    axes[1, 0].set_ylabel('Max Median Drift (%)')
    axes[1, 0].set_title('Relative Median Drift')
    axes[1, 0].axhline(y=50, color='r', linestyle='--', label='50% threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Maximum deviation in scaled space
    axes[1, 1].plot(drift_df['step'], 
                    drift_df['params_outside_3iqr'].apply(lambda x: x['max_deviation']))
    axes[1, 1].set_xlabel('Optimization Step')
    axes[1, 1].set_ylabel('Max Deviation (# of IQRs)')
    axes[1, 1].set_title('Maximum Parameter Deviation')
    axes[1, 1].axhline(y=3, color='orange', linestyle='--', label='3 IQR (outlier)')
    axes[1, 1].axhline(y=5, color='r', linestyle='--', label='5 IQR (extreme)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: Which parameters are drifting most
    if param_names and len(drift_df) > 0:
        last_outliers = drift_df.iloc[-1]['params_outside_3iqr']['outlier_indices']
        axes[1, 2].bar(range(len(last_outliers)), last_outliers)
        axes[1, 2].set_xlabel('Outlier Rank')
        axes[1, 2].set_ylabel('Parameter Index')
        axes[1, 2].set_title('Which Parameters are Outliers')
    
    plt.tight_layout()
    plt.show()

