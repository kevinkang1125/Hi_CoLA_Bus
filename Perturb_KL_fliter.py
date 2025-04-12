import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# === Behavioural Model ===
class Behavioural(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class Policy(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

def perturb_add(pi, sigma=0.5):
    pi_perturbed = Policy()
    pi_perturbed.load_state_dict(pi.state_dict())
    with torch.no_grad():
        for param in pi_perturbed.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * sigma
                param.add_(noise)
    return pi_perturbed

def bernoulli_kl(q, p, eps=1e-8):
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).sum(dim=-1)

def kl_between_policies(pi_perturbed, pi, states):
    with torch.no_grad():
        probs_pi = pi(states)
        probs_pert = pi_perturbed(states)
        kl_vals = bernoulli_kl(probs_pert, probs_pi)
    return kl_vals.mean()

def get_kl_folder(kl_score):
    bins = np.arange(0, 0.5, 0.1)
    for start in bins:
        end = round(start + 0.1, 1)
        if start <= kl_score < end:
            return f"{start:.1f}-{end:.1f}"
    return None

# === Main ===
df = pd.read_csv("./traj.csv")
states = torch.tensor(df.iloc[:60000, :5].values, dtype=torch.float32)
behavior_pi = Behavioural()
behavior_pi.load_state_dict(torch.load("./Behavioural_model_2.pth"))

target_count = 200
bin_counts = {f"{round(start,1):.1f}-{round(start+0.1,1):.1f}": 0 for start in np.arange(0, 0.5, 0.1)}
bin_indices = {k: 0 for k in bin_counts.keys()}  # for naming files

sigma = 0.4

while not all(count >= target_count for count in bin_counts.values()):
    pi_perturbed = perturb_add(pi=behavior_pi, sigma=sigma)
    kl_score = kl_between_policies(pi_perturbed, behavior_pi, states).item()

    if kl_score < 0.5:
        folder = get_kl_folder(kl_score)
        if folder and bin_counts[folder] < target_count:
            save_path = f"./Policys/{folder}/"
            os.makedirs(save_path, exist_ok=True)
            filename = f"Perturbed_model_{bin_indices[folder]}.pth"
            torch.save(pi_perturbed, os.path.join(save_path, filename))
            bin_counts[folder] += 1
            bin_indices[folder] += 1
            print(f"Saved to {folder}/{filename} (KL={kl_score:.4f}) — count: {bin_counts[folder]}/{target_count}")
