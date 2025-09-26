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
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fix NumPy compatibility issue
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler



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
    
    return train_loader, val_loader, scaler, y.max()


# === IMPROVED MODEL ===
class Hi_CoLA_Net(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        
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



def train(features, labels, model, lr=5e-5, weight_decay=1e-5, epochs=2500, 
          verbose=False, logit=False, train_ratio=0.7, use_improvements=True):
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
    train_loader, val_loader, scaler,label_up = load_model_features_and_labels(
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
    improved_metrics = evaluate(features, labels, model)
    
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
    
    return model, train_losses, val_losses, scaler, improved_metrics['r2'], label_up


def evaluate(features, labels, model):
    """
    IMPROVED evaluation with proper preprocessing consistency
    """
    train_loader, val_loader, _,_ = load_model_features_and_labels(
        features, labels, train_ratio=0.8)
    
    # model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            y_true.extend(yb.squeeze().tolist())
            y_pred.extend(pred.squeeze().tolist())
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
    # plt.figure(figsize=(8, 6))
    # plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=30)
    
    # # Better axis limits
    # y_min, y_max = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    # margin = (y_max - y_min) * 0.1
    # plt.xlim(y_min - margin, y_max + margin)
    # plt.ylim(y_min - margin, y_max + margin)
    
    # plt.plot([y_min - margin, y_max + margin], [y_min - margin, y_max + margin], 
    #          'r--', label="Ideal (y = x)", linewidth=2)
    # plt.xlabel("Actual", fontsize=14)
    # plt.ylabel("Predicted", fontsize=14)
    # plt.title(f"Predicted vs Actual\nR² = {r2:.4f}, MAPE = {mape:.2f}%", fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.tight_layout()
    # plt.show()
    
    return {'mse': mse, 'mae': mae, 'r2': r2, 'mape': mape}




# STEP 1: Add this wrapper class to your file (copy-paste anywhere after your imports)

class HiCoLAWithScaler(nn.Module):
    """Wrapper that handles RobustScaler automatically"""
    def __init__(self, hi_cola_model, scaler):
        super().__init__()
        self.hi_cola = hi_cola_model
        
        if scaler is not None:
            self.register_buffer('scaler_center', torch.FloatTensor(scaler.center_))
            self.register_buffer('scaler_scale', torch.FloatTensor(scaler.scale_))
            self.has_scaler = True
        else:
            self.has_scaler = False
    
    def forward(self, raw_policy_params):
        if self.has_scaler:
            normalized_params = (raw_policy_params - self.scaler_center) / self.scaler_scale
            return self.hi_cola(normalized_params)
        else:
            return self.hi_cola(raw_policy_params)

