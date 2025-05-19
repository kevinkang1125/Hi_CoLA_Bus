from Behaviour_pi import Behavioural, Policy, Hi_CoLA
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

behavior_pi = Behavioural()
behavior_pi.load_state_dict(torch.load("./Behavioural_model.pth"))

base_pi = Behavioural()
base_pi.load_state_dict(torch.load("./Behavioural_model.pth"))
df = pd.read_csv("./traj.csv")
states = torch.tensor(df.iloc[:60000, :5].values, dtype=torch.float32)

input_dim = sum(p.numel() for p in behavior_pi.parameters())
model = Hi_CoLA(input_dim=input_dim)
model.load_state_dict(torch.load("./Hi_CoLA_net.pth"))
model.eval()

# Flatten the Behavioural model's parameters into a vector (input to KLRegressor)
def get_flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

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

# Setup optimizer to update the behavioural model's parameters
optimizer = torch.optim.Adam(behavior_pi.parameters(), lr=0.004)

# Target KL divergence we want to achieve
target_kl = torch.tensor([[1]], dtype=torch.float32)

# Optimization loop
loss_fn = nn.MSELoss()
loss_trace, kl_trace = [], []
kl_divs = []

for step in range(1200):
    optimizer.zero_grad()
    
    # Extract flattened weights from behavioural_model
    input_vector = get_flat_params(behavior_pi).unsqueeze(0)
    
    # Feed into the frozen KL regressor
    kl_output = model(input_vector)
    
    # Minimize difference to desired KL
    loss = loss_fn(kl_output, target_kl)
    loss.backward()
    optimizer.step()
    kl_div = kl_between_policies(behavior_pi, base_pi, states)
    # Logging
    loss_trace.append(loss.item())
    kl_trace.append(kl_output.item())
    kl_divs.append(kl_div.item())
    
    if step % 50 == 0 or step == 299:
        print(f"Step {step:03d} | Loss: {loss.item():.6f} | KL: {kl_output.item():.4f}")

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(loss_trace)
plt.xlabel("Step",fontsize = 20)
plt.ylabel("Loss",fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)


kl_divs = np.minimum(kl_divs,3)
plt.subplot(1, 2, 2)
plt.plot(kl_divs)
plt.xlabel("Step",fontsize = 20)
plt.ylabel("Kl_Divergence_Track",fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)
# # plt.tight_layout()
plt.show()
# print(kl_divs)
# torch.save(behavior_pi.state_dict(), "./Optimized_Behavioural_model.pth")