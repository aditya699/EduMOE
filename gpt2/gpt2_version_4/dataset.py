"""
Author : Aditya Bhatt 12:16PM
Please reach out to ab0358031@gmail.com for any issues and suggestions.
NOTE:

Notes on WikiTextDataset

Purpose:
- Converts raw text data into training-ready (x, y) pairs for next-token prediction.
- Implements the sliding window with stride method to efficiently use data.

Tokenization & Flattening:
- All texts are tokenized into integer IDs using GPT-2’s tokenizer.
- EOS (end-of-sequence) tokens are added between documents to separate them.
- The entire dataset is flattened into one long stream of tokens.

Sliding Window Chunking:
- Each training sample is a window of block_size + 1 tokens.
- Windows overlap by block_size - stride tokens to reduce data wastage.

Example:
    Tokens: [10, 20, 30, 40, 99]
    block_size=4 → windows of length 5
    x = [10, 20, 30, 40]
    y = [20, 30, 40, 99]

Why x = chunk[:-1] and y = chunk[1:]?
- This creates input-output pairs where the model sees the first N tokens (x) and predicts the next N tokens (y).
- Enforces the autoregressive training objective:
    L = -∑ₜ log P(xₜ | x₍ₜ₋₁₎)

Relation to Attention:
- During training, a causal mask ensures token t only attends to [1...t], never to the future.
- So for x = [10, 20, 30]:
    Prediction target is y = [20, 30, next].
    Internally, token 3's prediction (30 -> next) only attends to [10, 20, 30].

Batching:
- DataLoader stacks multiple (x, y) sequences into a batch.
- Shape:
    x_batch = [B, T]
    y_batch = [B, T]
- Loss is computed across all B * T predictions in parallel, then averaged.

Key Insight:
- Dataset construction ensures the model is always trained to predict “the next token given all previous tokens” — the foundation of GPT pretraining.


TODO:
1.Check for Dead imports
"""

from typing import List

import torch
from torch.utils.data import Dataset


class WikiTextDataset(Dataset):
    """Dataset with sliding-window chunking - EXACTLY YOUR ORIGINAL."""
    
    def __init__(self, texts: List[str], tokenizer, block_size: int = 512, stride: int = 512):
        self.block_size = block_size
        self.examples = []
        print("Tokenizing dataset with sliding windows...")
        
        # Always build a single concatenated token stream with EOS separators,
        # then take sliding windows across the whole stream. This keeps
        # train/val construction consistent and avoids tiny per-text fragments.
        window_len = block_size + 1
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        all_tokens: List[int] = []
        for text in texts:
            t = text.strip()
            if not t:
                continue
            all_tokens.extend(tokenizer.encode(t))
            if eos_id is not None:
                all_tokens.append(eos_id)
        
        if len(all_tokens) >= window_len:
            for i in range(0, len(all_tokens) - window_len + 1, stride):
                chunk = all_tokens[i:i + window_len]
                self.examples.append(torch.tensor(chunk, dtype=torch.long))
            if (len(all_tokens) - window_len) % stride != 0:
                tail = all_tokens[-window_len:]
                self.examples.append(torch.tensor(tail, dtype=torch.long))
        
        print(f"Total chunks created: {len(self.examples)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
