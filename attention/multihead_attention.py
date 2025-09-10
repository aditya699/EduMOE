import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads  # dimension per head

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_K = nn.Linear(embed_size, embed_size, bias=False)
        self.W_V = nn.Linear(embed_size, embed_size, bias=False)

        # Final linear projection
        self.W_O = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, X):
        """
        X: (batch, seq_len, embed_size)
        """
        batch_size, seq_len, _ = X.size()

        # Project into Q, K, V
        Q = self.W_Q(X)  # (batch, seq_len, embed_size)
        K = self.W_K(X)
        V = self.W_V(X)

        # Reshape into multiple heads: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 1: Compute scaled dot-product scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: (batch, num_heads, seq_len, seq_len)

        # Step 2: Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)

        # Step 3: Weighted sum with V
        out = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)

        # Step 4: Concatenate heads back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)

        # Step 5: Final projection
        out = self.W_O(out)  # (batch, seq_len, embed_size)

        return out, attn_weights

# ----------------------------
# Demo
# ----------------------------
batch_size = 1
seq_len = 4
embed_size = 8
num_heads = 2

# Fake input (4 tokens, each embed_size=8)
X = torch.randn(batch_size, seq_len, embed_size)

mha = MultiHeadAttention(embed_size, num_heads)
out, attn_weights = mha(X)

print("Attention weights shape:", attn_weights.shape)  # (batch, heads, seq_len, seq_len)
print("Output shape:", out.shape)                      # (batch, seq_len, embed_size)
