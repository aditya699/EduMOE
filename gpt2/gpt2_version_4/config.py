"""
Configuration for GPT-2 v4 training.

NOTE:
1.If you would use H100 or A100 please use bfloat16 instead of float16.

"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # ===== Model config (GPT-2 Medium scale) =====
    vocab_size: int = 50257                # Size of the vocabulary
    max_position_embeddings: int = 1024    # Maximum sequence length (context window)
    n_embd: int = 768                      # Embedding dimension (hidden size)
    n_layer: int = 12                      # Number of transformer layers
    n_head: int = 12                       # Number of attention heads
    dropout: float = 0.1                   # Dropout rate for regularization

    # ===== Training config =====
    batch_size: int = 4                    # Number of samples per batch
    gradient_accumulation_steps: int = 8   # Number of steps to accumulate gradients before optimizer step
    learning_rate: float = 1.5e-4          # Initial learning rate
    warmup_steps: int = 4000               # Number of steps for learning rate warmup
    max_steps: int = 300_000               # Total number of training steps
    max_epochs: Optional[int] = None       # Maximum number of epochs (None means unlimited)
    weight_decay: float = 0.01             # Weight decay for optimizer (L2 regularization)
    max_grad_norm: float = 1.0             # Maximum gradient norm for gradient clipping

    # ===== Data config =====
    block_size: int = 1024                 # Length of each training sample (in tokens)
    stride: int = 512                      # Stride for creating overlapping blocks from text

    # ===== Logging & Monitoring =====
    eval_interval: int = 5000              # Steps between evaluations on validation set
    log_interval: int = 50                 # Steps between logging training metrics
    save_interval: int = 10_000            # Steps between saving full model checkpoints
    light_save_interval: int = 2000        # Steps between saving lightweight checkpoints

    # ===== Experiment Tracking =====
    wandb_project: str = "gpt2-prodrun-v1"  # Weights & Biases project name
    wandb_name: Optional[str] = None                    # Optional run name for Weights & Biases
    use_wandb: bool = True                              # Whether to use Weights & Biases logging

    # ===== Mixed Precision =====
    amp_dtype: str = "float16"              # AMP (Automatic Mixed Precision) data type ("float16" or "bfloat16")


# Default config instance
default_config = TrainingConfig()
