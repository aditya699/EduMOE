import torch
import torch.nn as nn
try:
    from .attention import CausalSelfAttention  # type: ignore
    from .mlp import MLP  # type: ignore
except Exception:  # noqa: E722 - fallback when running as a script
    from attention import CausalSelfAttention  # type: ignore
    from mlp import MLP  # type: ignore


class TransformerBlock(nn.Module):
    """A single GPT-2 Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


