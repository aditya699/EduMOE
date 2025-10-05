import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    GPT-2 style feedforward network (MLP).
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = torch.nn.functional.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


