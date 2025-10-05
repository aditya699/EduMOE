"""
Author : Aditya Bhatt 12:16PM
Please reach out to ab0358031@gmail.com for any issues and suggestions.
NOTE:

Notes on BinaryTokenDataset

Purpose:
- Loads pre-tokenized binary data and creates training-ready (x, y) pairs for next-token prediction.
- Implements the sliding window with stride method to efficiently use data.

Binary Format:
- Tokens are stored as int32 (4 bytes each) in .bin files
- Memory-mapped for efficient access without loading entire dataset into RAM
- Pre-tokenized using custom tokenizer (stored in processed/tokenizer.json)

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
- Dataset construction ensures the model is always trained to predict "the next token given all previous tokens" — the foundation of GPT pretraining.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryTokenDataset(Dataset):
    """Dataset that loads pre-tokenized binary data with sliding-window chunking."""

    def __init__(self, bin_path: str, block_size: int = 512, stride: int = 512):
        """
        Args:
            bin_path: Path to .bin file containing int32 tokens
            block_size: Length of each training sequence
            stride: Step size between consecutive windows
        """
        self.block_size = block_size
        self.stride = stride

        # Memory-map the binary file for efficient access
        # dtype=np.int32 because tokens are stored as 4-byte integers
        self.data = np.memmap(bin_path, dtype=np.int32, mode='r')

        # Calculate number of chunks we can create
        window_len = block_size + 1
        if len(self.data) >= window_len:
            # Number of complete windows with stride
            self.n_chunks = (len(self.data) - window_len) // stride + 1

            # Check if we need an extra chunk for the tail
            if (len(self.data) - window_len) % stride != 0:
                self.n_chunks += 1
        else:
            self.n_chunks = 0

        print(f"Loaded binary dataset from {bin_path}")
        print(f"Total tokens: {len(self.data):,}")
        print(f"Total chunks created: {self.n_chunks:,}")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        """
        Returns:
            x: Input sequence [block_size]
            y: Target sequence (shifted by 1) [block_size]
        """
        window_len = self.block_size + 1

        # Handle last chunk specially (might be the tail)
        if idx == self.n_chunks - 1 and (len(self.data) - window_len) % self.stride != 0:
            # This is the tail chunk
            chunk = self.data[-window_len:]
        else:
            # Regular chunk with stride
            start_idx = idx * self.stride
            chunk = self.data[start_idx:start_idx + window_len]

        # Convert to torch tensor
        chunk = torch.from_numpy(chunk.astype(np.int64))

        # Create (x, y) pair
        x = chunk[:-1]  # First block_size tokens
        y = chunk[1:]   # Next block_size tokens (shifted by 1)

        return x, y
