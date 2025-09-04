import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------
# 1. Dataset
# ---------------------------
torch.manual_seed(42)
x = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(1)   # (200,1)
y = torch.sin(x)

# ---------------------------
# 2. RMSNorm definition
# ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # γ
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.scale * x / (rms + self.eps)

# ---------------------------
# 3. FFN with RMSNorm Pre-Norm
# ---------------------------
class ModernFFN(nn.Module):
    def __init__(self, d_model=1, d_ff=64, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),  # keep bias for toy task
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_normed = self.norm(x)
        return self.dropout(self.ffn(x_normed))

model = ModernFFN()

# ---------------------------
# 4. Loss + Optimizer
# ---------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------------------------
# 5. Training loop
# ---------------------------
epochs = 10000
losses = []

for epoch in range(epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ---------------------------
# 6. Plot predictions
# ---------------------------
with torch.no_grad():
    y_pred = model(x)

plt.figure(figsize=(8,5))
plt.scatter(x.numpy(), y.numpy(), label="True sin(x)", color="blue", alpha=0.5)
plt.plot(x.numpy(), y_pred.numpy(), label="FFN (RMSNorm)", color="red", linewidth=2)
plt.legend()
plt.show()
