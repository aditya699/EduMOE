# ---------------------------
# Feedforward Network (FFN) Example
# Task: Learn sin(x)
# ---------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Dataset
torch.manual_seed(42)
x = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(1)   # shape (200, 1)
y = torch.sin(x)

# 2. Define FFN
class SimpleFFN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleFFN()

# 3. Loss + Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 10000
losses = []

for epoch in range(epochs):
    # forward
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. Plot loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.show()

# 6. Plot predictions vs true
with torch.no_grad():
    y_pred = model(x)

plt.figure(figsize=(8,5))
plt.scatter(x.numpy(), y.numpy(), label="True sin(x)", color="blue", alpha=0.5)
plt.plot(x.numpy(), y_pred.numpy(), label="FFN predictions", color="red", linewidth=2)
plt.legend()
plt.show()
