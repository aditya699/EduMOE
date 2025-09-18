import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head masked (causal) self-attention for GPT-2.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by number of heads"

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Linear projections for Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Dropout (applied to attention weights and final projection)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: upper-triangular matrix of -inf
        mask = torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.max_position_embeddings, config.max_position_embeddings),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Linear projections → Q, K, V
        qkv = self.c_attn(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)  # each [B, T, C]

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention with causal masking for better stability
        # sdpa handles scaling, softmax, masking, and dropout internally in a numerically stable way
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )  # [B, nh, T, hd]

        # Recombine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection + dropout
        y = self.resid_dropout(self.c_proj(y))

        return y


