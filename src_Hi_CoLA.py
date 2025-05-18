import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from Behaviour_pi import Behavioural, Policy
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def logit(y, eps=1e-5):
    y = np.clip(y, eps, 1 - eps)
    return np.log(y / (1 - y))

def inv_logit(x):
    return 1 / (1 + np.exp(-x))

def load_model_features_and_labels(features, labels, train_ratio=0.7,logit=False):
    """_summary_

    Args:
        features (list): list of pertubated models's weights
        labels (): confidence lower bound of the perturbated models

    Returns:
        train_loader, test_loader: train and test dataloaders
        
    """
    policy_params = [] 
    for model_path in features:

        model = torch.load(model_path)

        # Flatten all parameters to 1D feature vector
        flat_params = []
        for param in model.parameters():
            flat_params.append(np.array(param.detach().cpu().tolist()).flatten())
        feature_vector = np.concatenate(flat_params)

        policy_params.append(feature_vector)
    features = np.array(policy_params)
    X = np.array(features)
    y = np.array(labels)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if logit:
        y_tensor = torch.tensor(logit(np.array(y)), dtype=torch.float32).unsqueeze(1)
    else:
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    # === Dataset and DataLoader ===
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    return train_loader, val_loader


# === Model ===
class Hi_CoLA_Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    
def train(features, labels, model, lr=1e-6, weight_decay=1e-4, epochs=1000, verbose=True,logit=False,train_ratio=0.7):
    ##Rememeber to define model input dimension:Hi_CoLA_Net(input_dim=X.shape[1])
    # === Load Features and Labels ===
    train_loader,val_loader = load_model_features_and_labels(features, labels,train_ratio,logit)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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
        if verbose:
            if epoch % 50 == 0 or epoch == epochs-1:
                print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        if epoch > 200:
            if np.var(train_losses[-20:]) < 1e-2*np.mean(train_losses[-20:]):
                # Early stopping condition
                print(f"Early stopping at epoch {epoch+1}")
                break
    if verbose:
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.title('Learning Curve', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()
    
    return model, train_losses, val_losses

def evaluate(features, labels, model, val_loader,logit=False):
    train_loader,val_loader = load_model_features_and_labels(features, labels)
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            y_true.extend(yb.squeeze().tolist())
            y_pred.extend(pred.squeeze().tolist())
    if logit:
        y_true = inv_logit(np.array(y_true))
        y_pred = inv_logit(np.array(y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

    print(f"\nEvaluation Metrics on Validation Set:")
    print(f"MSE  = {mse:.6f}")
    print(f"MAE  = {mae:.6f}")
    print(f"R²   = {r2:.4f}")
    print(f"MAPE = {mape:.2f}%")

    # === Scatter Plot ===
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([0.3, 0.5], [0.3, 0.5], 'r--', label="Ideal (y = x)")
    plt.xlabel("Actual", fontsize=14)
    plt.ylabel("Predicted", fontsize=14)
    plt.title(f"Validation: Predicted vs Actual\nMAPE = {mape:.2f}%")
    plt.xlim(0.3, 0.5)
    plt.ylim(0.3, 0.5)
    plt.legend()
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


