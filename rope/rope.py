import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Instead of adding sinusoidal position vectors (absolute),
    RoPE rotates the embedding pairs by an angle proportional
    to the token's position index.
    
    This rotation makes attention *relative-position aware*,
    which allows models to scale to much longer context windows
    more naturally than fixed absolute encodings.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        
        # Build frequency scales (like sinusoidal encodings).
        # inv_freq[i] = 1 / (base^(2i/dim))
        # These frequencies decide how fast/slow each embedding
        # dimension oscillates with position.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Register as buffer → moves with the model (GPU/CPU),
        # but does not get trained (fixed, like sinusoids).
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device=None):
        """
        Returns sine and cosine rotation angles for each position.
        
        Args:
            seq_len: number of tokens in the sequence
            device: CUDA/CPU
        Returns:
            sin, cos tensors of shape (seq_len, dim)
        """
        # Positions [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        
        # Compute outer product: (positions × frequencies)
        # Shape: (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Duplicate across two halves → (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Return rotation angles for each token & dimension
        return torch.sin(emb), torch.cos(emb)


def apply_rotary(x, sin, cos):
    """
    Apply RoPE rotation to a tensor of embeddings.
    
    Args:
        x:  (batch, seq_len, num_heads, head_dim) — queries/keys
        sin: (seq_len, head_dim) — sine angles
        cos: (seq_len, head_dim) — cosine angles
    
    Returns:
        Rotated embeddings, same shape as x.
    
    Why rotation?
    - Splitting embeddings into 2D pairs, then rotating them
      encodes position as an *angle*.
    - When queries and keys are compared in attention (dot product),
      the result depends only on their *relative rotation*,
      i.e., relative positions.
    - This enables smooth extrapolation to longer sequences,
      because distance (difference in angle) generalizes
      better than absolute indexes.
    """
    # Split last dimension into pairs: (x_even, x_odd)
    x1, x2 = x[..., ::2], x[..., 1::2]
    sin, cos = sin[..., ::2], cos[..., ::2]
    
    # Standard 2D rotation formula applied across all pairs:
    # (x', y') = (x*cos - y*sin,  x*sin + y*cos)
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return x_rotated
