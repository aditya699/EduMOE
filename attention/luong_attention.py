import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Luong "general" score uses a linear transform of encoder states
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def score(self, decoder_state, encoder_outputs):
        """
        decoder_state: (batch, hidden_size)
        encoder_outputs: (batch, seq_len, hidden_size)
        """
        # (batch, hidden_size) -> (batch, 1, hidden_size)
        dec = decoder_state.unsqueeze(1)
        # Apply linear to encoder outputs (general form)
        enc = self.W(encoder_outputs)  # (batch, seq_len, hidden_size)
        # Dot product along hidden dim
        scores = torch.bmm(dec, enc.transpose(1, 2))  # (batch, 1, seq_len)
        return scores.squeeze(1)  # (batch, seq_len)

    def forward(self, decoder_state, encoder_outputs):
        # Step 1: compute alignment scores
        scores = self.score(decoder_state, encoder_outputs)  # (batch, seq_len)

        # Step 2: softmax -> attention weights
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Step 3: weighted sum -> context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch, hidden_size)

        return context, attn_weights

# ----------------------------
# Demo
# ----------------------------
batch_size = 1
seq_len = 3
hidden_size = 8

# Fake encoder outputs (like for "I love India")
encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)

# Fake decoder hidden state (at some step t)
decoder_state = torch.randn(batch_size, hidden_size)

# Run attention
attention = LuongAttention(hidden_size)
context, attn_weights = attention(decoder_state, encoder_outputs)

print("Attention weights:", attn_weights)
print("Context vector:", context)
