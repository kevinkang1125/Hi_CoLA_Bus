import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Set Device (GPU if available) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load Data ===
df = pd.read_csv("traj.csv")

X = torch.tensor(df.iloc[:10000, :5].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:10000, 5:].values, dtype=torch.float32)

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=2)

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

# === Training Function with tqdm ===
def train_and_evaluate(hidden_dim=64, lr=0.001, num_epochs=50):
    model = MLP(input_dim=5, hidden_dim=hidden_dim, output_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss = criterion(pred, yb)
                epoch_val_loss += val_loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))

    return train_losses, val_losses

# === Run Training ===
train_losses, val_losses = train_and_evaluate()

# === Plotting ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss (MLP with GPU and tqdm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

