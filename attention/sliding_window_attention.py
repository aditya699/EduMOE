import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        """
        B, N, D = x.shape
        H = self.num_heads
        d_head = D // H

        # Project to Q, K, V
        Q = self.q_proj(x).view(B, N, H, d_head).transpose(1, 2)  # (B, H, N, d_head)
        K = self.k_proj(x).view(B, N, H, d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)  # (B, H, N, N)

        # Create sliding window mask
        mask = torch.full((N, N), float("-inf"), device=x.device)
        for i in range(N):
            start = max(0, i - self.window_size)
            end = min(N, i + self.window_size + 1)
            mask[i, start:end] = 0

        attn_scores = attn_scores + mask  # apply local window mask
        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_probs, V)  # (B, H, N, d_head)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

# Example usage
x = torch.randn(2, 16, 64)  # batch=2, seq_len=16, embed_dim=64
attn = SlidingWindowAttention(embed_dim=64, num_heads=4, window_size=2)
y = attn(x)
print(y.shape)  # (2, 16, 64)
