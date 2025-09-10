import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_K = nn.Linear(embed_size, embed_size, bias=False)
        self.W_V = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, X):
        """
        X: (batch, seq_len, embed_size)
        """
        Q = self.W_Q(X)   # (batch, seq_len, embed_size)
        K = self.W_K(X)   # (batch, seq_len, embed_size)
        V = self.W_V(X)   # (batch, seq_len, embed_size)

        # Step 1: Compute similarity scores
        # QK^T -> (batch, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)

        # Step 2: Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # Step 3: Weighted sum with V
        out = torch.matmul(attn_weights, V)  # (batch, seq_len, embed_size)

        return out, attn_weights

# ----------------------------
# Demo
# ----------------------------
batch_size = 1
seq_len = 3
embed_size = 4

# Fake token embeddings ("I", "love", "India")
X = torch.randn(batch_size, seq_len, embed_size)

self_attn = SelfAttention(embed_size)
out, attn_weights = self_attn(X)

print("Attention weights:\n", attn_weights)
print("\nOutput embeddings:\n", out)
