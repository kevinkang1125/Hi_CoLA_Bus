import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Example: 1000 samples, each with 20 features, binary classification
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))  # 0 or 1

# Wrap data into a PyTorch dataset
dataset = TensorDataset(X, y)

# Split into training and test sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define a simple feedforward neural network
class Behavior_P(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Behavior_P, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

model = Behavior_P(input_dim=20, hidden_dim=64, output_dim=2)
criterion = nn.CrossEntropyLoss()  # use BCEWithLogitsLoss for binary output (1 neuron)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.2%}")

torch.save(model.state_dict(), 'mlp_model.pth')

