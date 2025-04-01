import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# === Set Device (GPU if available) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# === MLP Model ===
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# === RMSE Loss Function ===
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

# === Training + Evaluation Function ===
def train_and_evaluate(hidden_dim, lr, num_epochs=50):
    model = MLP(input_dim=5, hidden_dim=hidden_dim, output_dim=2).to(device)
    criterion = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    # Validation RMSE
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item()
    avg_val_rmse = total_loss / len(val_loader)
    return avg_val_rmse

# === Grid Search ===
hidden_dims = [16, 32, 64, 128]
learning_rates = [round(0.001 * i, 3) for i in range(1, 11)]  # 0.001 to 0.01
data_size = [10000, 20000, 30000, 40000, 50000]
results = []

print("\n=== Hyperparameter Search ===")
for l in data_size:
    for h in hidden_dims:
        for lr in learning_rates:
            # === Load Data ===
            df = pd.read_csv("traj.csv")
            X = torch.tensor(df.iloc[:l, :5].values, dtype=torch.float32)
            y = torch.tensor(df.iloc[:l, 5:].values, dtype=torch.float32)

            dataset = TensorDataset(X, y)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_set, val_set = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=2)
            val_rmse = train_and_evaluate(hidden_dim=h, lr=lr)
            print(f"Data_size:{l}, Hidden: {h}, LR: {lr:.3f} => Val RMSE: {val_rmse:.4f}")
            results.append((l, h, lr, val_rmse))
colums = ["Data_size", "Hidden", "Learning Rate", "Validation RMSE"]
# === Sort and show best results ===
results.sort(key=lambda x: x[3])
grid_track = pd.DataFrame(results, columns=colums)
grid_track.to_csv("grid_search_results.csv", index=False)
best = results[0]
print("\n✅ Best Config:")
print(f"Hidden: {best[0]}, LR: {best[1]:.3f}, Val RMSE: {best[2]:.4f}")
