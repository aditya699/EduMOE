"""
Author : Aditya Bhatt 12:16PM


NOTE:
1.The entire architecture sits here
"""

import torch
import torch.nn as nn
from .blocks import TransformerBlock

class GPT2Model(nn.Module):
    """
    GPT-2 language model (core architecture).

    Components:
    1. Token Embeddings (wte): map vocab IDs → dense vectors.
    2. Positional Embeddings (wpe): encode token order.
    3. Transformer Blocks (h): stack of identical blocks, each with:
        - LayerNorm → CausalSelfAttention → Residual Add
        - LayerNorm → MLP → Residual Add
    4. Final LayerNorm (ln_f): stabilize output before logits.
    5. LM Head (lm_head): projects hidden states back to vocab size
       for next-token prediction. Tied to token embeddings.

    Args:
        config: TrainingConfig or SimpleNamespace with attributes:
            - vocab_size, n_embd, n_layer, n_head, max_position_embeddings, dropout
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # === Embeddings ===
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)  # positional embeddings
        self.drop = nn.Dropout(config.dropout)

        # === Transformer stack ===
        # List of N identical blocks (depth = n_layer)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # === Final norm + head ===
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights (embedding matrix shared with output head)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass of GPT-2.

        Args:
            input_ids (torch.Tensor):
                [batch_size, seq_len] token IDs.
            labels (torch.Tensor, optional):
                [batch_size, seq_len] target IDs for loss computation.

        Returns:
            dict with:
                - logits: [B, T, vocab_size] prediction scores
                - loss:   scalar (if labels provided)
        """
        B, T = input_ids.size()

        # Position IDs: [0, 1, ..., T-1]
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        # === Embeddings ===
        tok_emb = self.wte(input_ids)   # [B, T, n_embd]
        pos_emb = self.wpe(pos)         # [1, T, n_embd]
        x = self.drop(tok_emb + pos_emb)

        # === Transformer blocks ===
        for block in self.h:
            x = block(x)

        # === Final norm + LM head ===
        x = self.ln_f(x)
        logits = self.lm_head(x)        # [B, T, vocab_size]

        # === Loss (optional) ===
        loss = None
        if labels is not None:
            # Shift for autoregressive loss:
            # Predict token t+1 given tokens ≤ t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {"logits": logits, "loss": loss}

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids)
                logits = outputs["logits"]
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


    
