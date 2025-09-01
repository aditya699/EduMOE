import torch
import torch.nn as nn
import math
from pydantic import BaseModel, Field


class PositionalEncodingConfig(BaseModel):
    """Configuration for sinusoidal positional encoding."""
    embedding_dim: int = Field(..., description="Embedding dimension (same as token embeddings)")
    max_len: int = Field(5000, description="Maximum sequence length (context window)")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, config: PositionalEncodingConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.embedding_dim
        max_len = config.max_len

        # Create matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Different frequencies for each dimension (0,2,4,...)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer → not learnable, moves with model to GPU
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch_size, seq_len, embedding_dim)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
