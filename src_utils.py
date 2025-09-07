import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from Behaviour_pi import Behavioural, Policy



def kl_between_policies(pi_perturbed, pi, states,eps=1e-8):
    with torch.no_grad():
        probs_pi = pi(states)
        probs_pert = pi_perturbed(states)
        p = torch.clamp(probs_pi, eps, 1 - eps)
        q = torch.clamp(probs_pert, eps, 1 - eps)
        # kl_vals = bernoulli_kl(probs_pert, probs_pi)
        kl_vals = (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).sum(dim=-1)
    return kl_vals.mean()

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


