from pydantic import BaseModel, Field
import torch
import torch.nn as nn

class EmbeddingConfig(BaseModel):
    """Configuration for token embeddings."""
    vocab_size: int = Field(..., description="Number of tokens in the vocabulary")
    embedding_dim: int = Field(..., description="Dimension of each embedding vector")

class TokenEmbedding(nn.Module):
    """Embedding layer for tokens."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Map token IDs to embedding vectors.
        Args:
            token_ids (Tensor): shape (batch_size, seq_len)
        Returns:
            Tensor: shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(token_ids)


