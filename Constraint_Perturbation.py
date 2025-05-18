import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from Behaviour_pi import Behavioural, Policy
from src_utils import kl_between_policies


def perturb_add(pi, sigma=0.4):
    pi_perturbed = Policy()
    pi_perturbed.load_state_dict(pi.state_dict())
    with torch.no_grad():
        for param in pi_perturbed.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * sigma
                param.add_(noise)
    return pi_perturbed


# === Main ===
df = pd.read_csv("./traj.csv")
states = torch.tensor(df.iloc[:60000, :5].values, dtype=torch.float32)
behavior_pi = Behavioural()
behavior_pi.load_state_dict(torch.load("./Behavioural_model_2.pth"))

target_count = 200


sigma = 0.4
count = 0

while not count >= target_count:
    pi_perturbed = perturb_add(pi=behavior_pi, sigma=sigma)
    kl_score = kl_between_policies(pi_perturbed, behavior_pi, states).item()

    if kl_score < 0.25:
        count += 1
        save_path = f"./Policys/"
        os.makedirs(save_path, exist_ok=True)
        filename = f"Perturbed_model_{count}.pth"
        torch.save(pi_perturbed.state_dict(), os.path.join(save_path, filename))
