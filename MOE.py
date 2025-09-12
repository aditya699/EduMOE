# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Loading

# %%
#Install Important Libs
# %pip install -q datasets transformers pydantic matplotlib


# %%
from typing import Optional, Literal
from pydantic import BaseModel, Field

# %% [markdown]
# # Environment Check before running the code

# %%
import importlib
from typing import Optional, Literal
from pydantic import BaseModel

# --- Safe optional imports ---
if importlib.util.find_spec("torch"):
    import torch
else:
    torch = None

if importlib.util.find_spec("datasets"):
    import datasets
else:
    datasets = None

if importlib.util.find_spec("transformers"):
    import transformers
else:
    transformers = None


class LibraryVersions(BaseModel):
    torch: str = " Not Installed"
    datasets: str = " Not Installed"
    transformers: str = " Not Installed"


class GPUInfo(BaseModel):
    device: Literal["GPU", "CPU", "No Device Detected"]
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    memory_gb: Optional[float] = None
    message: Optional[str] = None


class EnvironmentCheck:
    """Environment checker with Pydantic models."""

    def __init__(self) -> None:
        self.torch_version = torch.__version__ if torch else None
        self.datasets_version = datasets.__version__ if datasets else None
        self.transformers_version = transformers.__version__ if transformers else None

    def get_versions(self) -> LibraryVersions:
        """Return installed library versions as a Pydantic model."""
        return LibraryVersions(
            torch=self.torch_version or " Not Installed",
            datasets=self.datasets_version or " Not Installed",
            transformers=self.transformers_version or " Not Installed",
        )

    def check_gpu(self) -> GPUInfo:
        """Return GPU details as a Pydantic model."""
        if not torch:
            return GPUInfo(device="No Device Detected")

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GPUInfo(
                device="GPU",
                cuda_version=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(0),
                memory_gb=round(props.total_memory / (1024**3), 2),
            )
        return GPUInfo(device="CPU", message="No GPU detected")


# --- Usage for manual run ---
if __name__ == "__main__":
    env = EnvironmentCheck()
    print("Library Versions:", env.get_versions().model_dump())
    print("Device Info:", env.check_gpu().model_dump())


# %% [markdown]
# # Data Loading

# %%
from typing import Optional
from pydantic import BaseModel, Field
from datasets import load_dataset, DatasetDict


class DatasetConfig(BaseModel):
    """Configuration model for dataset loading."""
    dataset_name: str = Field(..., description="Name of the dataset to load (e.g., 'wikitext').")
    subset_name: str = Field(..., description="Subset/config name (e.g., 'wikitext-2-raw-v1').")


class DatasetLoader:
    """Wrapper class for loading HuggingFace datasets."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def load_data(self) -> DatasetDict:
        """Load dataset using HuggingFace."""
        dataset = load_dataset(self.config.dataset_name, self.config.subset_name)
        return dataset


config = DatasetConfig(dataset_name="wikitext", subset_name="wikitext-2-raw-v1")
loader = DatasetLoader(config)
dataset = loader.load_data()

print(dataset)
for i in range(10):
    print(f"Row {i}: {dataset['train'][i]['text']!r}")



# %% [markdown]
# # Environment Setup and Dataset Preparation

# %% [markdown]
# 🔹 Environment Setup
#
# Before training a language model, we must ensure the environment is ready:
#
# Libraries: PyTorch for neural networks, HuggingFace datasets for loading corpora, and transformers for tokenization.
#
# GPU Availability: Training transformers on CPU is not practical. We verified CUDA version, GPU type, and VRAM size.
#
# Reproducibility: Using a structured check (EnvironmentCheck), we can confirm dependencies and hardware setup are consistent across machines.
#
# 🔹 Dataset Preparation in LLM Training
#
# Large-scale models are trained on diverse, massive corpora:
#
# 🌍 Common Crawl → large-scale web scrape (cleaned and deduplicated).
#
# 📚 Books → public-domain and licensed.
#
# 📝 Wikipedia → factual, well-structured text.
#
# 💻 Code → GitHub, forums, Q&A.
#
# 🧪 Research Articles → scientific sources like arXiv and PubMed.
#
# Such datasets often reach billions to trillions of tokens. Preprocessing includes deduplication, filtering, and quality checks.
#
# 🔹 Our Choice for EduMoE
#
# For this educational project, we use:
#
# ✅ WikiText-2 (raw):
#
# ~2M tokens — small enough for fast experiments.
#
# Natural, factual prose similar to real-world corpora.
#
# Already split into train, validation, and test.
#
# Includes quirks like blank lines, which helps mimic real preprocessing challenges.
#
# This dataset is ideal for learning the mechanics of LLM training without overwhelming compute resources.

# %% [markdown]
# # Tokenization

# %%
from typing import List
from pydantic import BaseModel, Field
from transformers import AutoTokenizer


class TokenizerConfig(BaseModel):
    """Configuration for tokenizer setup."""
    tokenizer_name: str = Field(default="gpt2", description="Pretrained tokenizer to use.")
    add_special_tokens: bool = Field(default=False, description="Whether to add special tokens like BOS/EOS.")


class TokenizerWrapper:
    """Wrapper for HuggingFace GPT-2 tokenizer."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        # GPT-2 has no pad token by default → we assign EOS as pad for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str) -> List[int]:
        """Convert text → token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=self.config.add_special_tokens)

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs → text."""
        return self.tokenizer.decode(tokens)

    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size
        

# --- Usage ---
if __name__ == "__main__":
    config = TokenizerConfig(tokenizer_name="gpt2", add_special_tokens=False)
    tok = TokenizerWrapper(config)

    text = "EduMoE is our Mixture of Experts project."
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)

    print("Vocab size:", tok.vocab_size())
    print("Text:", text)
    print("Tokens:", tokens)
    print("Decoded:", decoded)


# %% [markdown]
# # Tokenize WikiText-2 with Our Classes

# %%
# 1. Load WikiText-2 (raw)
data_config = DatasetConfig(dataset_name="wikitext", subset_name="wikitext-2-raw-v1")
dataset_loader = DatasetLoader(data_config)
dataset = dataset_loader.load_data()

# 2. Load GPT-2 Byte-Level BPE tokenizer
tok_config = TokenizerConfig(tokenizer_name="gpt2", add_special_tokens=False)
tokenizer = TokenizerWrapper(tok_config)

# 3. Grab a non-empty sample from training set
sample_text = next(x["text"] for x in dataset["train"] if x["text"].strip() != "")

# 4. Encode → token IDs
token_ids = tokenizer.encode(sample_text)

# 5. Decode → back to text
decoded_text = tokenizer.decode(token_ids)

# 6. Print results
print("Original text:\n", sample_text[:200], "\n")
print("Token IDs (first 40):\n", token_ids[:40], "...\n")
print("Decoded text:\n", decoded_text[:200], "\n")
print("Char length vs Token length:", len(sample_text), "chars →", len(token_ids), "tokens")
print("Tokenizer vocab size:", tokenizer.vocab_size())


# %% [markdown]
# # Before implementing MOE , We will be implementing a decoder style transformer and then convert it into a MOE Type design
# # This is how a gpt-2 style model looks like
# ![Capture](images/Capture.PNG)

# %% [markdown]
# # Embedding
# ![emblookup](images/emblookup.PNG)
# ![learn_para](images/learn_para.PNG)

# %% [markdown]
# # We will using nn.embedding as a safe wrapper for lookup table and otherwise
# ![nn.emb.PNG](images/nn.emb.PNG)

# %%
from pydantic import BaseModel, Field
import torch
import torch.nn as nn

class EmbeddingConfig(BaseModel):
    """Configuration for token embeddings."""
    vocab_size: int = Field(..., description="Number of tokens in the vocabulary")
    embedding_dim: int = Field(..., description="Dimension of each embedding vector")

class TokenEmbedding(nn.Module):
    """Embedding layer for tokens."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Map token IDs to embedding vectors.
        Args:
            token_ids (Tensor): shape (batch_size, seq_len)
        Returns:
            Tensor: shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(token_ids)




# %% [markdown]
# # Connecting everything till now

# %%
# 1. Load WikiText-2 dataset
data_config = DatasetConfig(dataset_name="wikitext", subset_name="wikitext-2-raw-v1")
dataset_loader = DatasetLoader(data_config)
dataset = dataset_loader.load_data()

# 2. Load tokenizer (GPT-2 BBPE)
tok_config = TokenizerConfig(tokenizer_name="gpt2", add_special_tokens=False)
tokenizer = TokenizerWrapper(tok_config)

# 3. Get vocab size from tokenizer
vocab_size = tokenizer.vocab_size()
embedding_dim = 128   # keep small for demo, GPT-2 uses 768

# 4. Create embedding layer
embed_config = EmbeddingConfig(vocab_size=vocab_size, embedding_dim=embedding_dim)
embed_layer = TokenEmbedding(embed_config)

# 5. Take one sample from dataset (non-empty)
sample_text = next(x["text"] for x in dataset["train"] if x["text"].strip() != "")
print("Original text:", sample_text[:100], "\n")

# 6. Tokenize → IDs
token_ids = tokenizer.encode(sample_text)
token_tensor = torch.tensor([token_ids])  # add batch dim

print("Token IDs:", token_ids[:20], "...\n")

# 7. Pass through embeddings
vectors = embed_layer(token_tensor)
print("Embeddings shape:", vectors.shape)
print("First token vector (truncated):", vectors[0, 0, :10])


# %% [markdown]
# # positional embeddings
#
# # Formula
# ![pos_emb](images/pos_emb.PNG)
#
# # Benifits
# ![pos_emb_1](images/pos_emb_1.PNG)
#
# # the sinusoidal positional encoding (original Transformer, Vaswani et al. 2017) is not learnable.
#
# # Lately system use somehting known Rotary Position Embeddings (RoPE)

# %%
import torch
import torch.nn as nn
import math
from pydantic import BaseModel, Field


class PositionalEncodingConfig(BaseModel):
    """Configuration for sinusoidal positional encoding."""
    embedding_dim: int = Field(..., description="Embedding dimension (same as token embeddings)")
    max_len: int = Field(5000, description="Maximum sequence length (context window)")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, config: PositionalEncodingConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.embedding_dim
        max_len = config.max_len

        # Create matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Different frequencies for each dimension (0,2,4,...)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer → not learnable, moves with model to GPU
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch_size, seq_len, embedding_dim)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]



# %%
# Token embeddings first
embed_config = EmbeddingConfig(vocab_size=50257, embedding_dim=128)
embed_layer = TokenEmbedding(embed_config)

# Positional encodings
pos_config = PositionalEncodingConfig(embedding_dim=128, max_len=512)
pos_encoder = PositionalEncoding(pos_config)

# Example input: 1 sentence, 9 tokens
token_ids = torch.tensor([[796, 569, 18354, 7496, 17740, 6711, 796, 220, 198]])
token_vectors = embed_layer(token_ids)

print("Before Positional Encoding:", token_vectors[0, 0, :5])

# Add positional encodings
out = pos_encoder(token_vectors)

print("After Positional Encoding:", out[0, 0, :5])
print("Output shape:", out.shape)


# %% [markdown]
# # Before moving towards multihead attention and other forms of attention best is to reivse a simple feed forward network

# %%
import torch
import matplotlib.pyplot as plt

# generate data
torch.manual_seed(42)  # reproducibility
x = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(1)  # shape (200, 1)
y = torch.sin(x)  # target

# visualize
plt.scatter(x.numpy(), y.numpy(), label="True function")
plt.legend()
plt.show()


# %%
#Input layer → hidden layer (ReLU) → output layer

import torch.nn as nn

class SimpleFFN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # (1 -> 16) #Here there 16 neurons in the hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # (16 -> 1)
        )

    def forward(self, x):
        return self.net(x)



# %%
# create model
model = SimpleFFN(input_dim=1, hidden_dim=16, output_dim=1)

# pick a sample x
x_sample = torch.tensor([[1.0]])  # shape (1,1)
y_pred = model(x_sample)

print("Input:", x_sample.item())
print("Model output (untrained):", y_pred.item())


# %%
#Setup Loss function and Optimizer
import torch.optim as optim

# loss function
criterion = nn.MSELoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)


# %%
# training loop
epochs = 1000
for epoch in range(epochs):
    # forward pass
    y_pred = model(x)
    
    # loss
    loss = criterion(y_pred, y)
    
    # backward pass
    optimizer.zero_grad()   # reset gradients
    loss.backward()         # compute gradients
    optimizer.step()        # update weights
    
    # log every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# %%
# model predictions after training
with torch.no_grad():  # no gradients needed
    y_pred = model(x)

# plot
plt.figure(figsize=(8,5))
plt.scatter(x.numpy(), y.numpy(), label="True sin(x)", color="blue", alpha=0.5)
plt.plot(x.numpy(), y_pred.numpy(), label="FFN predictions", color="red", linewidth=2)
plt.legend()
plt.show()


# %% [markdown]
# # Full Code for an FFN

# %%
# ---------------------------
# Feedforward Network (FFN) Example
# Task: Learn sin(x)
# ---------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Dataset
torch.manual_seed(42)
x = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(1)   # shape (200, 1)
y = torch.sin(x)

# 2. Define FFN
class SimpleFFN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleFFN()

# 3. Loss + Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 10000
losses = []

for epoch in range(epochs):
    # forward
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. Plot loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.show()

# 6. Plot predictions vs true
with torch.no_grad():
    y_pred = model(x)

plt.figure(figsize=(8,5))
plt.scatter(x.numpy(), y.numpy(), label="True sin(x)", color="blue", alpha=0.5)
plt.plot(x.numpy(), y_pred.numpy(), label="FFN predictions", color="red", linewidth=2)
plt.legend()
plt.show()


# %% [markdown]
# # From Basic FFN to Modern FFN
#
# The above was a **basic FFN**. Lately, things have changed a bit — the **heart is still the same**, but modern architectures add some refinements:
#
# ---
#
# ###  Step A – RMSNorm (Pre-Norm)
# - Normalize the input **before** passing into the FFN.  
# - Use **RMSNorm** (lighter, used in *LLaMA* / *Mistral*) instead of LayerNorm.  
#
# ---
#
# ###  Step B – SwiGLU (Modern Gating)
# - Replace the plain MLP with **SwiGLU** for richer dynamics.  
# - Adjust hidden dimension properly so parameter count ≈ classic 4×.  
#
# ---
#
# ### Step C – Dropout (Two Places)
# - **Inside FFN** (after activation/gating).  
# - **After final projection** (residual dropout).  
#
# ---
#
# ###  Step D – DropPath (Optional, Training Stabilizer)
# - Randomly skip whole FFN blocks during training.  
# - Helps with regularization in very deep networks.  
#

# %% [markdown]
# # 🔹 RMSNorm (Pre-Norm) — Theory
#
# ### LayerNorm (baseline)
# $$
# \text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
# $$
#
# - Subtracts the mean $\mu$ and divides by standard deviation $\sigma$ across features.  
# - Has learnable parameters: $\gamma$ (scale) and $\beta$ (shift).  
# - Works well, but slightly heavier due to mean calculation.  
#
# ---
#
# ### RMSNorm (modern choice)
# $$
# \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
# $$
#
# where  
#
# $$
# \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
# $$
#
# - **No mean subtraction** (lighter than LayerNorm).  
# - Only rescales by Root Mean Square (RMS).  
# - Uses **scale parameter $\gamma$** (no shift $\beta$).  
# - Faster, simpler, and often just as effective.  
#
# ---
#
# ### Pre-Norm vs Post-Norm
# - **Post-Norm (old style, original Transformer):**  
# ## x → FFN → Norm → (add residual)  (Earlier)
# ### x → Norm → FFN → (add residual)  (Now)
#
# - **Pre-Norm helps stabilize very deep networks and avoids vanishing gradients.**
#
#
# ✅ Modern FFNs use **RMSNorm in Pre-Norm style**, making them lighter and more stable than the ori

# %%
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        # learnable scale parameter γ (same size as hidden dimension)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute RMS over last dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()  # shape: (batch, seq, 1)
        x_normed = x / (rms + self.eps)
        return self.scale * x_normed



# %% [markdown]
# # 🔹 Modern FFN (Step A: Add RMSNorm Pre-Norm)

# %%
import torch
import torch.nn as nn

# custom RMSNorm (from previous step)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # learnable γ

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_normed = x / (rms + self.eps)
        return self.scale * x_normed

# FFN with RMSNorm Pre-Norm
class ModernFFN(nn.Module):
    def __init__(self, d_model=16, d_ff=64, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)  # Pre-Norm
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),   # will replace with SwiGLU in Step B
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-Norm
        x_normed = self.norm(x)
        out = self.ffn(x_normed)
        return self.dropout(out)



# %% [markdown]
# # Train with RMS Norm

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------
# 1. Dataset
# ---------------------------
torch.manual_seed(42)
x = torch.linspace(-2*torch.pi, 2*torch.pi, 200).unsqueeze(1)   # (200,1)
y = torch.sin(x)

# ---------------------------
# 2. RMSNorm definition
# ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # γ
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.scale * x / (rms + self.eps)

# ---------------------------
# 3. FFN with RMSNorm Pre-Norm
# ---------------------------
class ModernFFN(nn.Module):
    def __init__(self, d_model=1, d_ff=64, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),  # keep bias for toy task
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_normed = self.norm(x)
        return self.dropout(self.ffn(x_normed))

model = ModernFFN()

# ---------------------------
# 4. Loss + Optimizer
# ---------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------------------------
# 5. Training loop
# ---------------------------
epochs = 10000
losses = []

for epoch in range(epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ---------------------------
# 6. Plot predictions
# ---------------------------
with torch.no_grad():
    y_pred = model(x)

plt.figure(figsize=(8,5))
plt.scatter(x.numpy(), y.numpy(), label="True sin(x)", color="blue", alpha=0.5)
plt.plot(x.numpy(), y_pred.numpy(), label="FFN (RMSNorm)", color="red", linewidth=2)
plt.legend()
plt.show()


# %% [markdown]
# # ⚠️ Why RMSNorm Fails in Our Tiny 1D Toy Example
#
# - In our setup, the input dimension is only **1**.  
# - RMSNorm computes:
#   $$
#   \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
#   $$
#   When $d=1$, this reduces to just:
#   $$
#   \text{RMS}(x) \approx |x|
#   $$
# - So RMSNorm normalizes every input to roughly **±1**, destroying the scale information the model needs to learn $\sin(x)$.  
#
# ---
#
# ### Additional Factors
# - We also disabled **biases** in Linear layers (bias=False) for modern FFNs, but in a **tiny regression task**, biases are actually needed to shift outputs.  
# - **Dropout** (even small) can harm training when the network and dataset are this small.  
#
# ---
#
# ### ✅ Takeaway
# - RMSNorm is designed for **high-dimensional embeddings** (e.g., 512, 1024, 4096 in Transformers).  
# - In such cases, the RMS over many features preserves enough information while stabilizing training.  
# - In our **1D toy case**, RMSNorm collapses variation → the model outputs a flat line.  
#
# 👉 For demonstration of RMSNorm, we should use **larger embedding sizes** (like in Transformers), not 1D toy regressions.
#

# %% [markdown]
# # Bahdanau Attention (2014) — Neural Machine Translation
#
# **Problem (pre-attention):**  
# In vanilla encoder–decoder RNNs, the entire input sequence was compressed into a **single fixed-length vector** (the final hidden state of the encoder).  
# - This worked for short sentences but failed badly for long ones.  
# - Information from early words was lost.  
# - Translation quality degraded as input length increased.  
#
# ---
#
# ### Core Idea (Bahdanau et al., 2014)
# Instead of relying only on the last encoder hidden state, let the decoder:
# 1. **Keep all encoder hidden states** \( h_1, h_2, …, h_T \).  
# 2. At each decoder step \( t \), **decide which encoder states to focus on** by computing an *alignment score*.  
# 3. Use these scores to build a weighted average of encoder states = **context vector** \( c_t \).  
# 4. Combine \( c_t \) with the decoder state \( s_t \) to make the next prediction.  
#
# This is called **soft alignment** because it distributes attention weights across all input words.
#
# ---
#
# ### Step-by-Step Math
#
# 1. **Alignment scores**  
# For each decoder step \( t \), compare decoder hidden state \( s_t \) with every encoder hidden state \( h_i \):  
#
# $$
# e_{t,i} = v_a^\top \tanh(W_s s_t + W_h h_i)
# $$  
#
# - \( W_s, W_h, v_a \) are learnable parameters.  
# - \( e_{t,i} \) = relevance of encoder word \( i \) when predicting the next output.  
#
# ---
#
# 2. **Attention weights (softmax normalization)**  
#
# $$
# \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
# $$  
#
# Now \(\alpha_{t,i}\) indicates *how much attention to give* to word \( i \).  
#
# ---
#
# 3. **Context vector**  
#
# $$
# c_t = \sum_i \alpha_{t,i} \cdot h_i
# $$  
#
# This is a dynamic summary of the source, focused on what matters for step \( t \).  
#
# ---
#
# 4. **Attended state**  
#
# $$
# \tilde{s}_t = \tanh(W_c [s_t ; c_t])
# $$  
#
# This merges “what the decoder knows so far” with “what it just focused on.”  
#
# ---
#
# 5. **Output prediction**  
#
# $$
# y_t = \text{softmax}(W_o \tilde{s}_t)
# $$  
#
# ---
#
# ### Intuition Recap
# - Encoder: produces a hidden state for each input word.  
# - Decoder step \( t \):  
#   - Computes alignment with each encoder state.  
#   - Builds a context vector as a weighted blend.  
#   - Uses this to predict the next word.  
#
# **Key difference from pre-attention:** Instead of a single static summary, the decoder dynamically decides *where to look* at every step.
#

# %%
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


# %% [markdown]
# # Luong (2015) Attention

# %% [markdown]
# # Bahdanau (2014) vs. Luong (2015) Attention
#
# ---
#
# ### 🌱 Bahdanau Attention (Additive Attention, 2014)
#
# **Key idea:** Use a small feed-forward neural network to compute *alignment scores*.  
# - He called this **additive attention**.
#
# **Steps:**
# 1. **Alignment score:**
#    $$
#    e_{t,i} = v_a^\top \tanh(W_s s_t + W_h h_i)
#    $$
#    - Learnable parameters: $W_s, W_h, v_a$  
#    - Nonlinear scoring function.
#
# 2. **Attention weights:**
#    $$
#    \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
#    $$
#
# 3. **Context vector:**
#    $$
#    c_t = \sum_i \alpha_{t,i} h_i
#    $$
#
# 4. **Combine context + decoder state:**
#    $$
#    \tilde{s}_t = \tanh(W_c [s_t ; c_t])
#    $$
#
# 5. **Predict next word:**
#    $$
#    y_t = \text{softmax}(W_o \tilde{s}_t)
#    $$
#
# ---
#
# ### 🌿 Luong Attention (Multiplicative Attention, 2015)
#
# **Key idea:** Simplify scoring with similarity measures.  
# - He called this **multiplicative attention**.  
# - Two main scoring variants:  
#   - **Dot:** $e_{t,i} = s_t^\top h_i$  
#   - **General:** $e_{t,i} = s_t^\top W h_i$
#
# **Steps:**
# 1. **Alignment score (dot/general):**
#    - Dot product or linear transform, no extra hidden layer.
#
# 2. **Attention weights:** *(same as Bahdanau)*  
#    $$
#    \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
#    $$
#
# 3. **Context vector:** *(same as Bahdanau)*  
#    $$
#    c_t = \sum_i \alpha_{t,i} h_i
#    $$
#
# 4. **Combine context + decoder state:**  
#    $$
#    \tilde{h}_t = \tanh(W_c [c_t ; s_t])
#    $$
#
# 5. **Predict next word:** *(same as Bahdanau)*  
#    $$
#    y_t = \text{softmax}(W_o \tilde{h}_t)
#    $$
#
# ---
#
# ### 🌍 Main Differences
#
# - **Scoring function:**  
#   - Bahdanau → additive MLP with nonlinearity.  
#   - Luong → dot product (fast) or linear “general” form.  
#
# - **Efficiency:**  
#   - Bahdanau slower, more parameters.  
#   - Luong lighter, faster, easier to scale.  
#
# - **Local vs. global:**  
#   - Luong also introduced **local attention**, focusing only on a window of encoder states, improving speed on long sequences.  
#
# **Shared parts:**  
# Both compute attention weights → context vector → combine with decoder state → predict next word.
#

# %% [markdown]
# # Global vs Local Attention (Luong, 2015)
#
# ---
#
# ### 🌍 Global Attention
# - At each decoder step \(t\), compute alignment scores between decoder state \(s_t\) and **all encoder states** \(h_1, h_2, …, h_S\).  
# - Apply softmax over the full sequence to get attention weights.  
# - Build context vector as weighted sum of **all encoder states**.  
# - Advantage: model can attend anywhere.  
# - Disadvantage: expensive for long sequences (\(O(S)\) per step) and focus may spread too thin.
#
# ---
#
# ### 🔍 Local Attention
# - Predict a **center position** \(p_t\) in the source sequence.  
#   - **Monotonic (local-m):**  
#     $$
#     p_t = t
#     $$
#   - **Predictive (local-p):**  
#     $$
#     p_t = S \cdot \sigma(v_p^\top \tanh(W_p s_t))
#     $$
#     where a small neural net predicts \(p_t\) from the decoder state.
#
# - Apply a **Gaussian prior** around \(p_t\):  
#   $$
#   G_{t,i} = \exp\!\Big(-\frac{(i - p_t)^2}{2\sigma^2}\Big)
#   $$
#
# - Final unnormalized attention:  
#   $$
#   \alpha_{t,i} \propto \exp(e_{t,i}) \cdot G_{t,i}
#   $$
#
# - Effect: model attends strongly near \(p_t\), but softly includes neighbors.  
# - Advantage: cheaper, sharper focus, better for long sequences.  
# - Disadvantage: risk of missing distant dependencies if \(p_t\) is predicted poorly.
#
# ---
#
# ### 🧠 Intuition
# - **Global:** a wide searchlight scanning the entire source sentence.  
# - **Local:** a focused spotlight predicted by the decoder, softly centered by a Gaussian curve.
#

# %%
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


# %% [markdown]
# # Self-Attention (Scaled Dot-Product Attention)
#
# ---
#
# ### Motivation
# - RNNs process tokens **sequentially** → slow, hard to parallelize.  
# - Self-Attention lets each token **attend to all others directly** → full parallelism on GPUs.  
#
# ---
#
# ### Step 1 — Queries, Keys, Values
# Each token embedding \(x_i \in \mathbb{R}^d\) is projected into:
#
# $$
# q_i = W_Q x_i, \quad k_i = W_K x_i, \quad v_i = W_V x_i
# $$
#
# - **Query**: what this token is asking for.  
# - **Key**: what this token can offer.  
# - **Value**: the actual information to share.  
#
# ---
#
# ### Step 2 — Similarity Scores
# Compare Query of token \(i\) against Keys of all tokens \(j\):
#
# $$
# \text{score}(i,j) = \frac{q_i^\top k_j}{\sqrt{d_k}}
# $$
#
# Scaling by \(\sqrt{d_k}\) prevents large dot products from destabilizing softmax.
#
# ---
#
# ### Step 3 — Attention Weights
# Convert scores into probabilities:
#
# $$
# \alpha_{i,j} = \frac{\exp(\text{score}(i,j))}{\sum_m \exp(\text{score}(i,m))}
# $$
#
# \(\alpha_{i,j}\) = how much token \(i\) attends to token \(j\).  
#
# ---
#
# ### Step 4 — Context Vector
# Each token builds a new representation as a weighted sum of Values:
#
# $$
# z_i = \sum_j \alpha_{i,j} v_j
# $$
#
# ---
#
# ### Step 5 — Matrix Form
# For the whole sequence (length \(n\)):
#
# $$
# Q = X W_Q, \quad K = X W_K, \quad V = X W_V
# $$
#
# $$
# \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
# $$
#
# - Output \(Z \in \mathbb{R}^{n \times d_v}\).  
# - Each row \(z_i\) = contextual embedding for token \(i\).  
#
# ---
#
#
#

# %% [markdown]
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class SelfAttention(nn.Module):
#     def __init__(self, embed_size):
#         super().__init__()
#         self.embed_size = embed_size
#         # Linear projections for Q, K, V
#         self.W_Q = nn.Linear(embed_size, embed_size, bias=False)
#         self.W_K = nn.Linear(embed_size, embed_size, bias=False)
#         self.W_V = nn.Linear(embed_size, embed_size, bias=False)
#
#     def forward(self, X):
#         """
#         X: (batch, seq_len, embed_size)
#         """
#         Q = self.W_Q(X)   # (batch, seq_len, embed_size)
#         K = self.W_K(X)   # (batch, seq_len, embed_size)
#         V = self.W_V(X)   # (batch, seq_len, embed_size)
#
#         # Step 1: Compute similarity scores
#         # QK^T -> (batch, seq_len, seq_len)
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)
#
#         # Step 2: Softmax to get attention weights
#         attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
#
#         # Step 3: Weighted sum with V
#         out = torch.matmul(attn_weights, V)  # (batch, seq_len, embed_size)
#
#         return out, attn_weights
#
# # ----------------------------
# # Demo
# # ----------------------------
# batch_size = 1
# seq_len = 3
# embed_size = 4
#
# # Fake token embeddings ("I", "love", "India")
# X = torch.randn(batch_size, seq_len, embed_size)
#
# self_attn = SelfAttention(embed_size)
# out, attn_weights = self_attn(X)
#
# print("Attention weights:\n", attn_weights)
# print("\nOutput embeddings:\n", out)
#

# %% [markdown]
# # Multi-Head Attention (MHA)
#
# ## 🎯 Motivation
#
# - **Single-head attention** learns only one type of relation between tokens
# - Language has **multiple simultaneous relations**:
#   - Subject–verb dependencies
#   - Object–verb relationships  
#   - Positional patterns
#   - Semantic roles
# - **Multi-Head Attention** = multiple attention "specialists" working in parallel
#
# ---
#
# ## 📐 Mathematical Framework
#
# ### Step 1: Per-Head Projections
# For each head $i = 1, \ldots, h$, project input embeddings $X \in \mathbb{R}^{n \times d}$:
#
# $$Q^i = XW_Q^i, \quad K^i = XW_K^i, \quad V^i = XW_V^i$$
#
# where $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d \times d_k}$ and typically $d_k = \frac{d}{h}$.
#
# ### Step 2: Scaled Dot-Product Attention
# Each head computes attention independently:
#
# $$Z^i = \text{softmax}\left(\frac{Q^i (K^i)^\top}{\sqrt{d_k}}\right)V^i$$
#
# Output: $Z^i \in \mathbb{R}^{n \times d_v}$ where $d_v = d_k$ typically.
#
# ### Step 3: Concatenation
# Concatenate all heads along the feature dimension:
#
# $$Z = [Z^1 \,;\, Z^2 \,;\, \ldots \,;\, Z^h] \in \mathbb{R}^{n \times (h \cdot d_v)}$$
#
# ### Step 4: Final Linear Projection
# Project back to embedding dimension:
#
# $$\text{MHA}(X) = ZW_O$$
#
# where $W_O \in \mathbb{R}^{(h \cdot d_v) \times d}$.
#
# ---
#
# ## 🔄 Architecture Flow
#
# ```
# Input Embeddings X ∈ ℝⁿˣᵈ
#          │
#          ▼
# ┌─────────────────────────┐
# │   Linear Projections    │
# │  W_Q^i, W_K^i, W_V^i    │
# │    for i = 1...h        │
# └─────────────────────────┘
#          │
#          ▼
# ┌──────────┬──────────┬──────────┐
# │  Head 1  │  Head 2  │  Head h  │
# │ ┌──────┐ │ ┌──────┐ │ ┌──────┐ │
# │ │Q¹K¹V¹│ │ │Q²K²V²│ │ │QʰKʰVʰ│ │
# │ └──────┘ │ └──────┘ │ └──────┘ │
# │ Attn(·)  │ Attn(·)  │ Attn(·)  │
# └──────────┴──────────┴──────────┘
#          │
#          ▼
#     Concatenate
#    [Z¹; Z²; ...; Zʰ]
#          │
#          ▼
# ┌─────────────────────────┐
# │  Final Linear Layer     │
# │        W_O              │
# └─────────────────────────┘
#          │
#          ▼
#    MHA Output ∈ ℝⁿˣᵈ
# ```
#
# ---
#
# ## 💡 Key Insights
#
# 1. **Parallel Processing**: Each head learns different attention patterns simultaneously
# 2. **Complementary Views**: Heads may specialize in:
#    - Syntactic dependencies (subject-verb)
#    - Semantic relationships (word meanings)
#    - Positional patterns (nearby tokens)
#    - Long-range dependencies
#
# 3. **Parameter Efficiency**: By using $d_k = d/h$, total parameters ≈ single-head attention
#
# 4. **Expressiveness**: $h$ heads can capture $h$ different types of relationships
#
# ---
#
# ## ⚙️ Implementation Notes
#
# - **Typical values**: $h = 8$ or $h = 12$ in practice
# - **Dimension splitting**: $d_k = d_v = d/h$ keeps parameter count manageable  
# - **Computational complexity**: $O(n^2 d)$ same as single-head, but with $h$× parallelization
# - **Memory**: Stores $h$ attention matrices of size $n \times n$
#
# ---
#
# *Next: Implement MHA in PyTorch with shape annotations and visualization!*

# %%
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


# %% [markdown]
# # Final Code for GPT-2
#
# # max_position_embeddings = 512 → theoretical maximum context length.
# # The embedding table is built for 512 positions, so the model could handle sequences that long.
#
# # block_size = 256 → practical training context length.
# # During training, you only ever feed chunks of 256 tokens.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Model config
    vocab_size: int = 50257  #vocab for tokenizer
    max_position_embeddings: int = 512
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 8
    dropout: float = 0.1
    
    # Training config
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_epochs: int = 10
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data config
    block_size: int = 256  # sequence length for training
    
    # Logging
    eval_interval: int = 500
    log_interval: int = 100
    save_interval: int = 1000


class WikiTextDataset(Dataset):
    """Dataset for WikiText-2 with proper tokenization and chunking."""
    
    def __init__(self, texts: List[str], tokenizer, block_size: int = 256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize all texts and concatenate
        print("Tokenizing dataset...")
        all_tokens = []
        
        for text in texts:
            if text.strip():  # Skip empty lines
                tokens = tokenizer.encode(text.strip())
                all_tokens.extend(tokens)
                all_tokens.append(tokenizer.eos_token_id)  # Add EOS between articles
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Dataset has {len(self.tokens)} tokens")
        
        # Calculate number of samples
        self.num_samples = max(0, len(self.tokens) - block_size)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get a chunk of tokens
        chunk = self.tokens[idx:idx + self.block_size + 1]
        
        # Input and target (shifted by 1)
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y


class TrainingManager:
    """Manages the training process."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and prepare data
        self.train_loader, self.val_loader = self._prepare_data()
        
        # Initialize model
        model_config = self._create_model_config()
        self.model = GPT2Model(model_config).to(self.device)
        print(f"Model has {self.model.get_num_params():,} parameters")
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def _create_model_config(self):
        """Create model configuration from training config."""
        from types import SimpleNamespace
        return SimpleNamespace(
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_inner=None,
            dropout=self.config.dropout,
            layer_norm_epsilon=1e-5,
            use_bias=True
        )
    
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare WikiText-2 dataset."""
        print("Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Create datasets
        train_texts = [item['text'] for item in dataset['train'] if item['text'].strip()]
        val_texts = [item['text'] for item in dataset['validation'] if item['text'].strip()]
        
        train_dataset = WikiTextDataset(train_texts, self.tokenizer, self.config.block_size)
        val_dataset = WikiTextDataset(val_texts, self.tokenizer, self.config.block_size)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'ln', 'norm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return optim.AdamW(optimizer_groups, lr=self.config.learning_rate)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return 1.0
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch) -> float:
        """Single training step."""
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        outputs = self.model(x, labels=y)
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def evaluate(self) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x, labels=y)
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit evaluation for faster training
                if num_batches >= 50:
                    break
        
        return total_loss / num_batches
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
    
    def generate_sample(self, prompt: str = "The", max_tokens: int = 50) -> str:
        """Generate a sample text during training."""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                do_sample=True
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        
        start_time = time.time()
        accumulated_loss = 0
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss
                
                # Update parameters every gradient_accumulation_steps
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.step += 1
                    
                    # Logging
                    if self.step % self.config.log_interval == 0:
                        avg_loss = accumulated_loss / self.config.log_interval
                        self.train_losses.append(avg_loss)
                        
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        
                        print(f"Step {self.step:5d} | "
                              f"Epoch {epoch:2d} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {lr:.2e} | "
                              f"Time: {elapsed:.1f}s")
                        
                        accumulated_loss = 0
                    
                    # Evaluation
                    if self.step % self.config.eval_interval == 0:
                        val_loss = self.evaluate()
                        self.val_losses.append(val_loss)
                        
                        print(f"Validation loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(f"best_model_step_{self.step}.pt")
                            print(f"New best model saved! Val loss: {val_loss:.4f}")
                        
                        # Generate sample
                        sample = self.generate_sample("The future of artificial intelligence")
                        print(f"Sample: {sample[:100]}...")
                        print("-" * 80)
                    
                    # Save checkpoint
                    if self.step % self.config.save_interval == 0:
                        self.save_checkpoint(f"checkpoint_step_{self.step}.pt")
                    
                    # Early stopping
                    if self.config.max_steps and self.step >= self.config.max_steps:
                        print(f"Reached max steps ({self.config.max_steps})")
                        return
        
        print("Training completed!")
        self.save_checkpoint("final_model.pt")
    
    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(12, 5))
        
        # Training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Steps (x100)')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses)
        plt.title('Validation Loss')
        plt.xlabel('Evaluations')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# Simplified GPT-2 model (use your full implementation)
class GPT2Model(nn.Module):
    """Simplified GPT-2 for training demo."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final norm and head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        b, t = input_ids.size()
        pos = torch.arange(0, t, device=input_ids.device).unsqueeze(0)
        
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {'logits': logits, 'loss': loss}
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, do_sample=True):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings)).view(1, 1, config.max_position_embeddings, config.max_position_embeddings))
    
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = torch.nn.functional.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ================================
# Main Training Script
# ================================

def main():
    """Main training function."""
    # Create training configuration
    config = TrainingConfig(
        # Model config (small for quick training)
        n_embd=256,
        n_layer=6,
        n_head=8,
        
        # Training config
        batch_size=4,  # Small batch for demo
        learning_rate=1e-4,
        max_epochs=3,
        warmup_steps=500,
        max_steps=20000,  # Limit for demo
        gradient_accumulation_steps=2,
        
        # Data config
        block_size=128,  # Smaller sequences for faster training
        
        # Logging
        eval_interval=200,
        log_interval=50,
        save_interval=500
    )
    
    # Create trainer and start training
    trainer = TrainingManager(config)
    trainer.train()
    
    # Plot results
    trainer.plot_losses()
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_val_loss = trainer.evaluate()
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Generate some samples
    print("\nGenerated Samples:")
    for prompt in ["The", "In the future", "Artificial intelligence"]:
        sample = trainer.generate_sample(prompt, max_tokens=30)
        print(f"Prompt: '{prompt}' -> {sample}")


if __name__ == "__main__":
    main()

# %%
import jupytext

# Load the notebook
nb = jupytext.read("MOE.ipynb")

# Convert and write to .py
jupytext.write(nb, "MOE.py")


# %%
