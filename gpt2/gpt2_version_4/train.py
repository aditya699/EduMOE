"""
Author : Aditya Bhatt 12:16PM
Please reach out to ab0358031@gmail.com for any issues and suggestions.
NOTE:
1.We are keeping tokenizer parallelization as false since we will run it in RUNPOD machines , sometimes we might see an issue.
(For local machines, we can keep it true)

2.The following code will run on RTX 5090 GPU.

3.We are using AMP 

AMP = Automatic Mixed Precision, a feature in PyTorch (torch.amp or torch.cuda.amp) that lets your model train using both:

FP16 (half precision, 16-bit floats) for most operations.

FP32 (full precision, 32-bit floats) for critical operations where accuracy matters.

4.We are going to stick to max_setps not epoochs



TODO:
1.Check for Dead imports

"""

# ==========================
# Standard Library Imports
# ==========================
import math
import os
from typing import Tuple

# ==========================
# Third-Party Libraries
# ==========================
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import trainer and config (pkg or script mode)
try:
    from .config import TrainingConfig  # type: ignore
    from .trainer import TrainingManager  # type: ignore
except Exception:  # noqa: E722 - fallback for script execution
    from config import TrainingConfig  # type: ignore
    from trainer import TrainingManager  # type: ignore
def main():
    # Enable TF32 for improved stability/perf on Ampere+ GPUs
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    config = TrainingConfig()
    print("GPT-2 v4 Modular Training")
    print("=" * 60)
    print(f"Model: {config.n_layer}L-{config.n_embd}d-{config.n_head}h")
    print(f"Context: {config.block_size}")
    print(f"Batch size: {config.batch_size} (x{config.gradient_accumulation_steps} grad accum)")
    print(f"Max steps: {config.max_steps}")
    print(f"AMP dtype: {config.amp_dtype}")
    print("=" * 60)

    trainer = TrainingManager(config)
    trainer.train()

    print("\nFinal Evaluation:")
    final_val_loss = trainer.evaluate()
    final_ppl = math.exp(min(final_val_loss, 10))
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final perplexity: {final_ppl:.2f}")

    print("\nGenerated Samples:")
    for prompt in ["The", "In the future", "Artificial intelligence"]:
        sample = trainer.generate_sample(prompt, max_tokens=30)
        print(f"Prompt: '{prompt}' -> {sample}")


if __name__ == "__main__":
    main()

