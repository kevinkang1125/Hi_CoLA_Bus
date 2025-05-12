from Behaviour_pi import Behavioural, Policy, Hi_CoLA
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

behavior_pi = Behavioural()
behavior_pi.load_state_dict(torch.load("./Behavioural_model.pth"))
input_dim = sum(p.numel() for p in behavior_pi.parameters())
model = Hi_CoLA(input_dim=input_dim)
model.load_state_dict(torch.load("./Hi_CoLA_net.pth"))
model.eval()

# Flatten the Behavioural model's parameters into a vector (input to KLRegressor)
def get_flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

# Setup optimizer to update the behavioural model's parameters
optimizer = torch.optim.Adam(behavior_pi.parameters(), lr=0.04)

# Target KL divergence we want to achieve
target_kl = torch.tensor([[1]], dtype=torch.float32)

# Optimization loop
loss_fn = nn.MSELoss()
loss_trace, kl_trace = [], []

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
    
    # Logging
    loss_trace.append(loss.item())
    kl_trace.append(kl_output.item())
    
    if step % 50 == 0 or step == 299:
        print(f"Step {step:03d} | Loss: {loss.item():.6f} | KL: {kl_output.item():.4f}")

plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
plt.plot(loss_trace)
# plt.title("Loss")
plt.xlabel("Step",fontsize = 20)
plt.ylabel("Loss",fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)



plt.subplot(1, 2, 2)
plt.plot(kl_trace)
# plt.axhline(target_kl.item(), linestyle='--', color='r', label='Target KL')
# plt.title("Confidence Lowerbound")
plt.xlabel("Step",fontsize = 20)
plt.ylabel("Predicted Confidence Lowerbound",fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)

# # plt.tight_layout()
plt.show()
torch.save(behavior_pi.state_dict(), "./Optimized_Behavioural_model.pth")