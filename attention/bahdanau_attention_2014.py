import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, attn_size):
        super().__init__()
        # Linear layers for computing alignment scores
        self.W_s = nn.Linear(hidden_size, attn_size)   # for decoder state
        self.W_h = nn.Linear(hidden_size, attn_size)   # for encoder states
        self.v_a = nn.Linear(attn_size, 1, bias=False) # score to scalar

    def forward(self, decoder_state, encoder_outputs):
        """
        decoder_state: (batch, hidden_size) at time t
        encoder_outputs: (batch, seq_len, hidden_size) for all encoder steps
        """
        # (batch, 1, hidden_size)
        dec = decoder_state.unsqueeze(1)

        # Project both
        dec_proj = self.W_s(dec)                        # (batch, 1, attn_size)
        enc_proj = self.W_h(encoder_outputs)            # (batch, seq_len, attn_size)

        # Broadcast addition + tanh
        scores = self.v_a(torch.tanh(dec_proj + enc_proj))  # (batch, seq_len, 1)

        # Drop the last dim
        scores = scores.squeeze(-1)  # (batch, seq_len)

        # Softmax over seq_len → attention weights
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum of encoder outputs → context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs) 
        context = context.squeeze(1)  # (batch, hidden_size)

        return context, attn_weights

# ----------------------------
# Demo
# ----------------------------
batch_size = 1
seq_len = 3
hidden_size = 8
attn_size = 6

# Fake encoder outputs (like for "I love India")
encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)

# Fake decoder hidden state (at some step t)
decoder_state = torch.randn(batch_size, hidden_size)

# Run attention
attention = BahdanauAttention(hidden_size, attn_size)
context, attn_weights = attention(decoder_state, encoder_outputs)

print("Attention weights:", attn_weights)
print("Context vector:", context)
